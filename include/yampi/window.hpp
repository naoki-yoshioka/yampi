#ifndef PASTEL_WINDOW_HPP
# define PASTEL_WINDOW_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/is_same.hpp>
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
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif


namespace yampi
{
  template <typename Value>
  class window
  {
    MPI_Win mpi_win_;
    Value* base_ptr_;

   public:
    window() : mpi_win_(MPI_WIN_NULL), base_ptr_(nullptr) { }
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
      : mpi_win_(std::move(other.mpi_win_)),
        base_ptr_(std::move(other.base_ptr_))
    { other.mpi_win_ = MPI_WIN_NULL; other.base_ptr_ = nullptr; }

    window& operator=(window&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_win_ = std::move(other.mpi_win_);
        base_ptr_ = std::move(other.base_ptr_);
        other.mpi_win_ = MPI_WIN_NULL;
        other.base_ptr_ = nullptr;
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

    window(MPI_Win const mpi_win, ::yampi::environment const& environment)
      : mpi_win_(mpi_win), base_ptr_(get_base_ptr(mpi_win, environment))
    { }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, MPI_INFO_NULL, communicator, environment)),
        base_ptr_(YAMPI_addressof(*first))
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           Value>::value),
        "Value must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    template <typename ContiguousIterator>
    window(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::information const& information,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
      : mpi_win_(create(first, last, information.mpi_info(), communicator, environment)),
        base_ptr_(YAMPI_addressof(*first))
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           Value>::value),
        "Value must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    // TODO: implement other constructors

   private:
    Value* get_base_ptr(MPI_Win const mpi_win, ::yampi::environment const& environment) const
    {
      Value* result;
      int exists_base_ptr;
      int const error_code
        = MPI_Win_get_attr(mpi_win, MPI_WIN_BASE, result, YAMPI_addressof(exists_base_ptr));
      return error_code == MPI_SUCCESS
        ? (static_cast<bool>(exists_base_ptr) ? result : nullptr)
        : throw ::yampi::error(error_code, "yampi::window::get_base_ptr", environment);
    }

    template <typename ContiguousIterator>
    MPI_Win create(
      ContiguousIterator const first, ContiguousIterator const last,
      MPI_Info mpi_info, ::yampi::communicator const& communicator,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           Value>::value),
        "Value must be tha same to value_type of ContiguousIterator");
      assert(last >= first);

      MPI_Win result;
      int const error_code
        = MPI_Win_create(
            YAMPI_addressof(*first),
            (::yampi::addressof(*last, environment) - ::yampi::addressof(*first, environment)).mpi_address(),
            sizeof(Value), mpi_info, communicator.mpi_comm(),
            YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::window::create", environment);
    }

   public:
    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Win const mpi_win, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = mpi_win;
      base_ptr_ = get_base_ptr(mpi_win, environment);
    }

    template <typename ContiguousIterator>
    void reset(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = create(first, last, MPI_INFO_NULL, communicator, environment);
      base_ptr_ = YAMPI_addressof(*first);
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
      base_ptr_ = YAMPI_addressof(*first);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(YAMPI_addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window::free", environment);
    }

    // TODO: implement attributes and info

    MPI_Win const& mpi_win() const { return mpi_win_; }
    Value const* base() const { return base_ptr_; }

    void swap(window& other)
      BOOST_NOEXCEPT_IF(
        ::yampi::utility::is_nothrow_swappable<MPI_Win>::value
        and ::yampi::utility::is_nothrow_swappable<Value*>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
      swap(base_ptr_, other.mpi_ptr_);
    }
  };

  template <typename Value>
  inline void swap(::yampi::window<Value>& lhs, ::yampi::window<Value>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same
# undef YAMPI_remove_cv

#endif

