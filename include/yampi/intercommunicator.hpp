#ifndef YAMPI_INTERCOMMUNICATOR_HPP
# define YAMPI_INTERCOMMUNICATOR_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/has_nothrow_constructor.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/group.hpp>
# include <yampi/information.hpp>
# include <yampi/color.hpp>
# include <yampi/split_type.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_default_constructible std::is_nothrow_default_constructible
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_nothrow_default_constructible boost::has_nothrow_default_constructor
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class intercommunicator
    : public ::yampi::communicator_base
  {
    typedef ::yampi::communicator_base base_type;

   public:
    intercommunicator() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_default_constructible<base_type>::value)
      : base_type()
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    intercommunicator(intercommunicator const&) = delete;
    intercommunicator& operator=(intercommunicator const&) = delete;
# else
   private:
    intercommunicator(intercommunicator const&);
    intercommunicator& operator=(intercommunicator const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    intercommunicator(intercommunicator&&) = default;
    intercommunicator& operator=(intercommunicator&&) = default;
#   endif
    ~intercommunicator() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    //using base_type::base_type;
    explicit intercommunicator(MPI_Comm const& mpi_comm)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type(mpi_comm)
    { }

    intercommunicator(intercommunicator const& other, ::yampi::environment const& environment)
      : base_type(other, environment)
    { }

# if MPI_VERSION >= 3
    intercommunicator(
      intercommunicator const& other, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : base_type(other, information, environment)
    { }
# endif

    intercommunicator(
      intercommunicator const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
      : base_type(other, group, environment)
    { }

    intercommunicator(
      intercommunicator const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
      : base_type(other, color, key, environment)
    { }

# if MPI_VERSION >= 3
    intercommunicator(
      intercommunicator const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : base_type(other, split_type, key, information, environment)
    { }

    intercommunicator(
      intercommunicator const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
      : base_type(other, split_type, key, ::yampi::information(), environment)
    { }
# endif

    intercommunicator(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment)
      : base_type(create(local_communicator, local_leader, peer_communicator, remote_leader, tag, environment))
    { }

   private:
    MPI_Comm create(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Intercomm_create(
            local_communicator.mpi_comm(), local_leader.mpi_rank(),
            peer_communicator.mpi_comm(), remote_leader.mpi_rank(),
            tag.mpi_tag(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::intercommunicator::create", environment);
    }

   public:
    using base_type::reset;

    void reset(
      ::yampi::communicator const& local_communicator, ::yampi::rank const local_leader,
      ::yampi::communicator const& peer_communicator, ::yampi::rank const remote_leader,
      ::yampi::tag const tag, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = create(local_communicator, local_leader, peer_communicator, remote_leader, tag, environment);
    }

    void merge(::yampi::communicator& communicator, ::yampi::environment const& environment) const
    { merge(communicator, true, environment); }

    void merge(
      ::yampi::communicator& communicator, bool const is_higher_rank_preferred,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Intercomm_merge(mpi_comm_, static_cast<int>(is_higher_rank_preferred), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::intercommunicator::merge", environment);

      communicator.reset(result, environment);
    }

    int remote_size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_remote_size(mpi_comm_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::intercommunicator::remote_size", environment);
    }

    void remote_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Comm_remote_group(mpi_comm_, YAMPI_addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::intercommunicator::remote_group", environment);

      group.reset(mpi_group, environment);
    }
  };

  inline void swap(::yampi::intercommunicator& lhs, ::yampi::intercommunicator& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_nothrow_default_constructible

#endif

