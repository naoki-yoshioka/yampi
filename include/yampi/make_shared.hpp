#ifndef YAMPI_MAKE_SHARED_HPP
# define YAMPI_MAKE_SHARED_HPP

# include <type_traits>
# include <utility>
# include <memory>

# include <mpi.h>

# include <yampi/detail/mpi_delete.hpp>


namespace yampi
{
  template <typename T, typename... Arguments>
  inline typename std::enable_if< not std::is_array<T>::value, std::shared_ptr<T> >::type
  make_shared(Arguments&&... arguments)
  {
    T* base_ptr;
    int const error_code
      = MPI_Alloc_mem(static_cast<MPI_Aint>(sizeof(T)), MPI_INFO_NULL, YAMPI_addressof(base_ptr));
    T* ptr = ::new(base_ptr) T(std::forward<Arguments>(arguments)...);
    return std::shared_ptr<T>(ptr, ::yampi::detail::mpi_delete<T>());
  }
}


#endif

