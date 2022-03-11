#ifndef YAMPI_COMMUNICATOR_BASE_HPP
# define YAMPI_COMMUNICATOR_BASE_HPP

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
# include <yampi/group.hpp>
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
  class communicator_base
  {
   protected:
    MPI_Comm mpi_comm_;

   public:
    communicator_base() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(MPI_COMM_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    communicator_base(communicator_base const&) = delete;
    communicator_base& operator=(communicator_base const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    communicator_base(communicator_base const&);
    communicator_base& operator=(communicator_base const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    communicator_base(communicator_base&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Comm>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Comm>::value)
      : mpi_comm_(std::move(other.mpi_comm_))
    { other.mpi_comm_ = MPI_COMM_NULL; }

    communicator_base& operator=(communicator_base&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Comm>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Comm>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        if (mpi_comm_ != MPI_COMM_NULL and mpi_comm_ != MPI_COMM_WORLD and mpi_comm_ != MPI_COMM_SELF)
          MPI_Comm_free(YAMPI_addressof(mpi_comm_));
        mpi_comm_ = std::move(other.mpi_comm_);
        other.mpi_comm_ = MPI_COMM_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

   protected:
    ~communicator_base() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      MPI_Comm_free(YAMPI_addressof(mpi_comm_));
    }

   public:
    explicit communicator_base(MPI_Comm const& mpi_comm)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Comm>::value)
      : mpi_comm_(mpi_comm)
    { }

    communicator_base(communicator_base const& other, ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, environment))
    { }

# if MPI_VERSION >= 3
    communicator_base(
      communicator_base const& other, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(duplicate(other, information, environment))
    { }
# endif

    communicator_base(
      communicator_base const& other, ::yampi::group const& group,
      ::yampi::environment const& environment)
      : mpi_comm_(create(other, group, environment))
    { }

    communicator_base(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, color, key, environment))
    { }

# if MPI_VERSION >= 3
    communicator_base(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, information, environment))
    { }

    communicator_base(
      communicator_base const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
      : mpi_comm_(split(other, split_type, key, ::yampi::information(), environment))
    { }
# endif

   private:
    MPI_Comm duplicate(communicator_base const& other, ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code = MPI_Comm_dup(other.mpi_comm(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::duplicate", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm duplicate(
      communicator_base const& other, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_dup_with_info(other.mpi_comm(), information.mpi_info(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::duplicate", environment);
    }
# endif

    MPI_Comm create(
      communicator_base const& other, ::yampi::group const& group,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_create(other.mpi_comm(), group.mpi_group(), YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::create", environment);
    }

    MPI_Comm split(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split(other.mpi_comm(), color.mpi_color(), key, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::split", environment);
    }

# if MPI_VERSION >= 3
    MPI_Comm split(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment) const
    {
      MPI_Comm result;
      int const error_code
        = MPI_Comm_split_type(
            other.mpi_comm(), split_type.mpi_split_type(), key, information.mpi_info(),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::split", environment);
    }
# endif

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Comm const& mpi_comm, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = mpi_comm;
    }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(communicator_base&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_comm_ = std::move(other.mpi_comm_);
      other.mpi_comm_ = MPI_COMM_NULL;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void reset(communicator_base const& other, ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = duplicate(other, environment);
    }

    void reset(communicator_base const& other, ::yampi::group const& group, ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = create(other, group, environment);
    }

    void reset(
      communicator_base const& other, ::yampi::color const color, int const key,
      ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, color, key, environment);
    }

# if MPI_VERSION >= 3
    void reset(
      communicator_base const& other, ::yampi::split_type const split_type,
      int const key, ::yampi::information const& information,
      ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, split_type, key, information, environment);
    }

    void reset(
      communicator_base const& other, ::yampi::split_type const split_type, int const key,
      ::yampi::environment const& environment)
    {
      if (this == YAMPI_addressof(other))
        return;

      free(environment);
      mpi_comm_ = split(other, split_type, key, ::yampi::information(), environment);
    }
# endif

    void free(::yampi::environment const& environment)
    {
      if (mpi_comm_ == MPI_COMM_NULL or mpi_comm_ == MPI_COMM_WORLD or mpi_comm_ == MPI_COMM_SELF)
        return;

      int const error_code = MPI_Comm_free(YAMPI_addressof(mpi_comm_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::free", environment);
    }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_comm_ == MPI_COMM_NULL))
    { return mpi_comm_ == MPI_COMM_NULL; }

    bool operator==(communicator_base const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_comm_ == other.mpi_comm_; }

    int size(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_size(mpi_comm_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::communicator_base::size", environment);
    }

    ::yampi::rank rank(::yampi::environment const& environment) const
    {
      int mpi_rank;
      int const error_code = MPI_Comm_rank(mpi_comm_, YAMPI_addressof(mpi_rank));
      return error_code == MPI_SUCCESS
        ? ::yampi::rank(mpi_rank)
        : throw ::yampi::error(error_code, "yampi::communicator_base::rank", environment);
    }

    void group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Comm_group(mpi_comm_, YAMPI_addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::group", environment);

      group.reset(mpi_group, environment);
    }

# if MPI_VERSION >= 3
    void set_information(::yampi::information const& information, ::yampi::environment const& environment) const
    {
      int const error_code = MPI_Comm_set_info(mpi_comm_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::set_information", environment);
    }

    void get_information(::yampi::information& information, ::yampi::environment const& environment) const
    {
      MPI_Info mpi_info;
      int const error_code = MPI_Comm_get_info(mpi_comm_, YAMPI_addressof(mpi_info));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::communicator_base::get_information", environment);
      information.reset(mpi_info, environment);
    }
# endif

    bool is_intercommunicator(::yampi::environment const& environment) const
    {
      int result;
      int const error_code = MPI_Comm_test_inter(mpi_comm_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? static_cast<bool>(result)
        : throw ::yampi::error(error_code, "yampi::communicator_base::is_intercommunicator", environment);
    }

    MPI_Comm const& mpi_comm() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_comm_; }

    void swap(communicator_base& other) BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Comm>::value)
    {
      using std::swap;
      swap(mpi_comm_, other.mpi_comm_);
    }
  }; // class communicator_base

  inline void swap(::yampi::communicator_base& lhs, ::yampi::communicator_base& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  inline bool is_valid_rank(
    ::yampi::rank const& rank, ::yampi::communicator_base const& communicator,
    ::yampi::environment const& environment)
  {
    return rank >= ::yampi::rank(0)
      and rank < ::yampi::rank(communicator.size(environment));
  }
} // namespace yampi


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif // YAMPI_COMMUNICATOR_BASE_HPP
