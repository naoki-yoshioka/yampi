#ifndef YAMPI_DETAIL_MPI_DELETE_HPP
# define YAMPI_DETAIL_MPI_DELETE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_convertible.hpp>
#   include <boost/type_traits/is_void.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_convertible std::is_convertible
#   define YAMPI_is_void std::is_void
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_convertible boost::is_convertible
#   define YAMPI_is_void boost::is_void
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  namespace detail
  {
    template <typename T>
    struct mpi_delete
    {
# ifndef BOOST_NO_CXX11_DEFAULT_FUNCTIONS
      BOOST_CONSTEXPR mpi_delete() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULT_FUNCTIONS
      BOOST_CONSTEXPR mpi_delete() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULT_FUNCTIONS

      template <typename U, typename = typename YAMPI_enable_if<YAMPI_is_convertible<U*, T*>::value>::type>
      mpi_delete(mpi_delete<U> const&) BOOST_NOEXCEPT_OR_NOTHROW
      { }

      void operator()(T* ptr) const
      {
        static_assert(not YAMPI_is_void<T>::value, "can't delete pointer to incomplete type");
        static_assert(sizeof(T) > 0, "can't delete pointer to incomplete type");
        ptr->~T();
        MPI_Free_mem(ptr);
      }
    };

    template <typename T>
    struct mpi_delete<T[]>
    {
# ifndef BOOST_NO_CXX11_DEFAULT_FUNCTIONS
      BOOST_CONSTEXPR mpi_delete() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULT_FUNCTIONS
      BOOST_CONSTEXPR mpi_delete() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULT_FUNCTIONS

      template <typename U, typename = typename YAMPI_enable_if<YAMPI_is_convertible<U(*)[], T(*)[]>::value>::type>
      mpi_delete(mpi_delete<U[]> const&) BOOST_NOEXCEPT_OR_NOTHROW
      { }

      template <typename U>
      typename YAMPI_enable_if<YAMPI_is_convertible<U(*)[], T(*)[]>::value, void>::type
      operator()(U* ptr) const
      {
        static_assert(sizeof(T) > 0, "can't delete pointer to incomplete type");
        MPI_Free_mem(ptr);
      }
    };
  }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_is_void
# undef YAMPI_is_convertible
# undef YAMPI_enable_if

#endif

