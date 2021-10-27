#ifndef YAMPI_MAKE_SHARED_HPP
# define YAMPI_MAKE_SHARED_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_SMART_PTR
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/utility/enable_if.hpp>
#     include <boost/type_traits/is_array.hpp>
#     include <boost/type_traits/integral_constant.hpp>
#   endif
#   include <utility>
#   include <memory>

#   include <mpi.h>

#   include <yampi/detail/mpi_delete.hpp>

#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     define YAMPI_enable_if std::enable_if
#     define YAMPI_is_array std::is_array
#   else
#     define YAMPI_enable_if boost::enable_if_c
#     define YAMPI_is_array boost::is_array
#   endif


namespace yampi
{
#   if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
  template <typename T, typename... Arguments>
  inline typename YAMPI_enable_if< not YAMPI_is_array<T>::value, std::shared_ptr<T> >::type
  make_shared(Arguments&&... arguments)
  {
    T* base_ptr;
    int const error_code
      = MPI_Alloc_mem(static_cast<MPI_Aint>(sizeof(T)), MPI_INFO_NULL, YAMPI_addressof(base_ptr));
    T* ptr = ::new(base_ptr) T(std::forward<Arguments>(arguments)...);
    return std::shared_ptr<T>(ptr, ::yampi::detail::mpi_delete<T>());
  }
#   endif // !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
}


#   undef YAMPI_is_array
#   undef YAMPI_enable_if
# endif // BOOST_NO_CXX11_SMART_PTR

#endif

