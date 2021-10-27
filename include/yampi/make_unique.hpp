#ifndef YAMPI_MAKE_UNIQUE_HPP
# define YAMPI_MAKE_UNIQUE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_SMART_PTR
#   include <cstddef>
#   include <algorithm>
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/utility/enable_if.hpp>
#     include <boost/type_traits/is_array.hpp>
#     include <boost/type_traits/remove_extent.hpp>
#     include <boost/type_traits/integral_constant.hpp>
#   endif
#   include <utility>
#   include <memory>

#   include <mpi.h>

#   include <yampi/detail/mpi_delete.hpp>
#   include <yampi/detail/is_bounded_array.hpp>
#   include <yampi/detail/is_unbounded_array.hpp>

#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     define YAMPI_enable_if std::enable_if
#     define YAMPI_is_array std::is_array
#     define YAMPI_remove_extent std::remove_extent
#   else
#     define YAMPI_enable_if boost::enable_if_c
#     define YAMPI_is_array boost::is_array
#     define YAMPI_remove_extent boost::remove_extent
#   endif


namespace yampi
{
#   if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
  template <typename T, typename... Arguments>
  inline typename YAMPI_enable_if<
    not YAMPI_is_array<T>::value,
    std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >
  >::type
  make_unique(Arguments&&... arguments)
  {
    T* base_ptr;
    int const error_code
      = MPI_Alloc_mem(static_cast<MPI_Aint>(sizeof(T)), MPI_INFO_NULL, YAMPI_addressof(base_ptr));
    T* ptr = ::new(base_ptr) T(std::forward<Arguments>(arguments)...);
    return std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >(ptr, ::yampi::detail::mpi_delete<T>());
  }
#   endif // !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)

  template <typename T>
  inline typename YAMPI_enable_if<
    ::yampi::detail::is_unbounded_array<T>::value,
    std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >
  >::type
  make_unique(std::size_t const size)
  {
    typedef YAMPI_remove_extent<T>::type value_type;
    value_type* base_ptr;
    int const error_code
      = MPI_Alloc_mem(
          static_cast<MPI_Aint>(size) * static_cast<MPI_Aint>(sizeof(value_type)),
          MPI_INFO_NULL, YAMPI_addressof(base_ptr));
    std::fill(base_ptr, base_ptr + size, value_type());
    return std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >(base_ptr, ::yampi::detail::mpi_delete<T>());
  }

#   if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#     ifndef BOOST_NO_CXX11_DELETE_FUNCTIONS
  template <typename T, typename... Arguments>
  inline typename YAMPI_enable_if<::yampi::detail::is_bounded_array<T>::value>::type
  make_unique(Arguments&&... arguments) = delete;
#     else // BOOST_NO_CXX11_DELETE_FUNCTIONS
  template <typename T, typename... Arguments>
  inline typename YAMPI_enable_if<::yampi::detail::is_bounded_array<T>::value>::type
  make_unique(Arguments&&... arguments);
#     endif // BOOST_NO_CXX11_DELETE_FUNCTIONS
#   endif // !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
}


#   undef YAMPI_remove_extent
#   undef YAMPI_is_array
#   undef YAMPI_enable_if
# endif // BOOST_NO_CXX11_SMART_PTR

#endif

