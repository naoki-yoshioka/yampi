#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/request.hpp>
# include <yampi/group.hpp>
# include <yampi/tag.hpp>
# include <yampi/information.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  struct world_communicator_t { };
  struct self_communicator_t { };

  class communicator
  {
    MPI_Comm mpi_comm_;

   public:
    communicator() : mpi_comm_(MPI_COMM_NULL) { }
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    communicator(communicator const&) = delete;
    communicator& operator=(communicator const&) = delete;
# else
   private:
    communicator(communicator const&);
    communicator& operator=(communicator const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    communicator(communicator&& other)
      : mpi_comm_(std::move(other.mpi_comm_))
    { other.mpi_comm_ = MPI_COMM_NULL; }

    communicator& operator=(communicator&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_comm_ = std::move(other.mpi_comm_);
        other.mpi_comm_ = MPI_COMM_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~communicator() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      MPI_Comm_free(YAMPI_addressof(mpi_comm_));
    }

    explicit communicator(MPI_Comm const mpi_comm)
      : mpi_comm_(mpi_comm)
    { }

    explicit communicator(::yampi::world_communicator_t const)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit communicator(::yampi::self_communicator_t const)
      : mpi_comm_(MPI_COMM_SELF)
    { }

    communicator(communicator const& other, ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, environment))
    { }

# if MPI_VERSION >= 3
    communicator(
      communicator const& other, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, information, environment))
    { }

    communicator(
      communicator const& other, ::yampi::request& request,
      ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, request, environment))
    { }
# endif

    communicator(
      communicator const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
      : mpi_comm_(create(other, group, environment))
    { }

# if MPI_VERSION >= 3
    communicator(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag,
      ::yampi::environment const& environment)
      : mpi_comm_(create(other, group, tag, environment))
    { }
# endif

    // TODO: Implement MPI_Comm_split/MPI_Comm_split_type constructors

   private:
    MPI_Comm duplicate(communicator const& other, ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code = MPI_Comm_dup(other.mpi_comm(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::communicator::duplicate", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm duplicate(
      communicator const& other, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_dup_with_info(
            other.mpi_comm(), information.mpi_info(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::communicator::duplicate", environment);
    }

    MPI_Comm duplicate(
      communicator const& other, ::yampi::request& request,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      MPI_Request mpi_request;
      int const error_code
        = MPI_Comm_idup(other.mpi_comm(), YAMPI_addressof(result), YAMPI_addressof(mpi_request));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "yampi::communicator::duplicate", environment);

      request.mpi_request(mpi_request);
      return result;
    }
# endif

    MPI_Comm create(
      communicator const& other, ::yampi::group const& group,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_create(other.mpi_comm(), group.mpi_group(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::create", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm create(
      communicator const& other, ::yampi::group const& group, ::yampi::tag const tag, ::yampi::environment const& environment) const
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
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Comm const mpi_comm, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = mpi_comm;
    }

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

    void reset(communicator const& other, ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = duplicate(other, environment);
    }

# if MPI_VERSION >= 3
    void reset(communicator const& other, ::yampi::request& request, ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = duplicate(other, request, environment);
    }
# endif

    void reset(
      communicator const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = create(other, group, environment);
    }

# if MPI_VERSION >= 3
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

    // TODO: Implement MPI_Comm_split/MPI_Comm_split_type constructors

    void free(::yampi::environment const& environment)
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      int const error_code = MPI_Comm_free(YAMPI_addressof(mpi_comm_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::free", environment);
    }


    bool is_null() const { return mpi_comm_ == MPI_COMM_NULL; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::size", environment);
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, YAMPI_addressof(mpi_rank));
      return error_code == MPI_SUCCESS
        ? ::yampi::rank(mpi_rank)
        : throw ::yampi::error(error_code, "yampi::communicator::size", environment);
    }

    void group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code
        = MPI_Comm_group(mpi_comm_, YAMPI_addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator::group", environment);

      group.reset(mpi_group, environment);
    }

# if MPI_VERSION >= 3
    void set_information(
      ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      int const error_code = MPI_Comm_set_info(mpi_comm_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "yampi::communicator::set_information", environment);
    }

    void get_information(
      ::yampi::information& information, ::yampi::environment const& environment) const
    {
      MPI_Info mpi_info;
      int const error_code = MPI_Comm_get_info(mpi_comm_, YAMPI_addressof(mpi_info));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "yampi::communicator::get_information", environment);
      information.reset(mpi_info, environment);
    }
# endif

    MPI_Comm const& mpi_comm() const { return mpi_comm_; }
    void mpi_comm(MPI_Comm const& comm) { mpi_comm_ = comm; }

    void swap(communicator& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Comm>::value)
    {
      using std::swap;
      swap(mpi_comm_, other.mpi_comm_);
    }
  };

  inline void swap(::yampi::communicator& lhs, ::yampi::communicator& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  inline bool is_valid_rank(
    ::yampi::rank const rank, ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    return rank >= ::yampi::rank(0)
      and rank < ::yampi::rank(communicator.size(environment));
  }
}


# undef YAMPI_addressof

#endif

