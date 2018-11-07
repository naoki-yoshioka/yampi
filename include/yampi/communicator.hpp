#ifndef YAMPI_COMMUNICATOR_HPP
# define YAMPI_COMMUNICATOR_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
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
# include <yampi/color.hpp>
# include <yampi/split_type.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
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

  class communicator
  {
    MPI_Comm mpi_comm_;

   public:
    communicator()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(MPI_COMM_NULL)
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

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    communicator(communicator&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Comm>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Comm>::value)
      : mpi_comm_(std::move(other.mpi_comm_))
    { other.mpi_comm_ = MPI_COMM_NULL; }

    communicator& operator=(communicator&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Comm>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Comm>::value)
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
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(mpi_comm)
    { }

    explicit communicator(::yampi::world_communicator_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(MPI_COMM_WORLD)
    { }

    explicit communicator(::yampi::self_communicator_t const)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
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

    communicator(
      communicator const& other, ::yampi::color const color, ::yampi::rank const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, color, key, environment))
    { }

# if MPI_VERSION >= 3
    communicator(
      communicator const& other, ::yampi::split_type const split_type,
      ::yampi::rank const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, information, environment))
    { }

    communicator(
      communicator const& other, ::yampi::split_type const split_type, ::yampi::rank const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, ::yampi::information(), environment))
    { }
# endif

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

    MPI_Comm split(
      communicator const& other, ::yampi::color const color, ::yampi::rank const key,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split(
            other.mpi_comm(), color.mpi_color(), key.mpi_rank(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::split", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm split(
      communicator const& other, ::yampi::split_type const split_type,
      ::yampi::rank const key, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split_type(
            other.mpi_comm(), split_type.mpi_split_type(), key.mpi_rank(), information.mpi_info(),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator::split", environment);
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


    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_comm_ == MPI_COMM_NULL))
    { return mpi_comm_ == MPI_COMM_NULL; }

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

    MPI_Comm const& mpi_comm() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_comm_; }
    void mpi_comm(MPI_Comm const& comm)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable<MPI_Comm>::value)
    { mpi_comm_ = comm; }

    void swap(communicator& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Comm>::value)
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
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

