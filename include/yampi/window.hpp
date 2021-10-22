#ifndef YAMPI_WINDOW_HPP
# define YAMPI_WINDOW_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/remove_cv.hpp>
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

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_remove_cv boost::remove_cv
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

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  class window
  {
    MPI_Win mpi_win_;

   public:
    window()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_(MPI_WIN_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    window(window const&) = delete;
    window& operator=(window const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    window(window const&);
    window& operator=(window const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    window(window&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_(std::move(other.mpi_win_))
    { other.mpi_win_ = MPI_WIN_NULL; }

    window& operator=(window&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_win_ = std::move(other.mpi_win_);
        other.mpi_win_ = MPI_WIN_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~window() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(YAMPI_addressof(mpi_win_));
    }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, MPI_INFO_NULL, communicator, environment))
    { assert(last >= first); }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, information.mpi_info(), communicator, environment))
    { assert(last >= first); }

   private:
    template <typename ContiguousIterator>
    MPI_Win create(
      ContiguousIterator const first, ContiguousIterator const last,
      MPI_Info const& mpi_info, ::yampi::communicator const& communicator,
      ::yampi::environment const& environment) const
    {
      assert(last >= first);

      using value_type = typename YAMPI_remove_cv<typename std::iterator_traits<ContiguousIterator>::value_type>::type;
      MPI_Win result;
      int const error_code
        = MPI_Win_create(
            YAMPI_addressof(*first),
            (::yampi::addressof(*last, environment) - ::yampi::addressof(*first, environment)).mpi_address(),
            sizeof(value_type), mpi_info, communicator.mpi_comm(),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::window::create", environment);
    }

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Win const& mpi_win, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = mpi_win;
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, MPI_INFO_NULL, communicator, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, information.mpi_info(), communicator, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(YAMPI_addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::free", environment);
    }

    void set_information(yampi::information const& information, yampi::environment const& environment) const
    {
      int const error_code = MPI_Win_set_info(mpi_win_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::information", environment);
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? yampi::information(result)
        : throw ::yampi::error(error_code, "yampi::window::information", environment);
    }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_win_ == MPI_WIN_NULL))
    { return mpi_win_ == MPI_WIN_NULL; }

    MPI_Win const& mpi_win() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_win_; }

    void swap(window& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
    }
  };

  inline void swap(::yampi::window& lhs, ::yampi::window& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_remove_cv

#endif

