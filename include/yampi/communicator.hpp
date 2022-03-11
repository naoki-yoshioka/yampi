#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

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
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/group.hpp>
# include <yampi/tag.hpp>
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
  struct world_communicator_t { };
  struct self_communicator_t { };

  namespace tags
  {
    inline BOOST_CONSTEXPR ::yampi::world_communicator_t world_communicator() { return ::yampi::world_communicator_t(); }
    inline BOOST_CONSTEXPR ::yampi::self_communicator_t self_communicator() { return ::yampi::self_communicator_t(); }
  }

  class communicator
    : public ::yampi::communicator_base
  {
    typedef ::yampi::communicator_base base_type;

   public:
    communicator() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_default_constructible<base_type>::value)
      : base_type()
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    communicator(communicator const&) = delete;
    communicator& operator=(communicator const&) = delete;
# else
   private:
    communicator(communicator const&);
    communicator& operator=(communicator const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    communicator(communicator&&) = default;
    communicator& operator=(communicator&&) = default;
#   endif
    ~communicator() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    //using base_type::base_type;
    explicit communicator(MPI_Comm const& mpi_comm)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type(mpi_comm)
    { }

    communicator(communicator const& other, ::yampi::environment const& environment)
      : base_type(other, environment)
    { }

# if MPI_VERSION >= 3
    communicator(
      communicator const& other, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : base_type(other, information, environment)
    { }
# endif

    communicator(
      communicator const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
      : base_type(other, group, environment)
    { }

    communicator(
      communicator const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
      : base_type(other, color, key, environment)
    { }

# if MPI_VERSION >= 3
    communicator(
      communicator const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : base_type(other, split_type, key, information, environment)
    { }

    communicator(
      communicator const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
      : base_type(other, split_type, key, ::yampi::information(), environment)
    { }
# endif

    explicit communicator(::yampi::world_communicator_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type(MPI_COMM_WORLD)
    { }

    explicit communicator(::yampi::self_communicator_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : base_type(MPI_COMM_SELF)
    { }

# if MPI_VERSION >= 3
    // only for intracommunicator
    communicator(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment)
      : base_type(create(other, group, tag, environment))
    { }
# endif

   private:
# if MPI_VERSION >= 3
    // only for intracommunicator
    MPI_Comm create(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_create_group(
            other.mpi_comm(), group.mpi_group(), tag.mpi_tag(),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::create", environment);
    }
# endif

   public:
    using base_type::reset;

    void reset(yampi::world_communicator_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = MPI_COMM_WORLD;
    }

    void reset(yampi::self_communicator_t const, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = MPI_COMM_SELF;
    }

# if MPI_VERSION >= 3
    // only for intracommunicator
    void reset(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = create(other, group, tag, environment);
    }
# endif
  };
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_nothrow_default_constructible

#endif

