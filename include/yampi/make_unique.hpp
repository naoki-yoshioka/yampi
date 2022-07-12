#ifndef YAMPI_MAKE_UNIQUE_HPP
# define YAMPI_MAKE_UNIQUE_HPP

# include <cstddef>
# include <algorithm>
# include <type_traits>
# include <utility>
# include <memory>

# include <mpi.h>

# include <yampi/detail/mpi_delete.hpp>
# include <yampi/detail/is_bounded_array.hpp>
# include <yampi/detail/is_unbounded_array.hpp>


namespace yampi
{
  template <typename T, typename... Arguments>
  inline typename std::enable_if<
    not std::is_array<T>::value,
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

  template <typename T>
  inline typename std::enable_if<
    ::yampi::detail::is_unbounded_array<T>::value,
    std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >
  >::type
  make_unique(std::size_t const size)
  {
    typedef std::remove_extent<T>::type value_type;
    value_type* base_ptr;
    int const error_code
      = MPI_Alloc_mem(
          static_cast<MPI_Aint>(size) * static_cast<MPI_Aint>(sizeof(value_type)),
          MPI_INFO_NULL, YAMPI_addressof(base_ptr));
    std::fill(base_ptr, base_ptr + size, value_type());
    return std::unique_ptr< T, ::yampi::detail::mpi_delete<T> >(base_ptr, ::yampi::detail::mpi_delete<T>());
  }

  template <typename T, typename... Arguments>
  inline typename std::enable_if<::yampi::detail::is_bounded_array<T>::value>::type
  make_unique(Arguments&&... arguments) = delete;
}


#endif

