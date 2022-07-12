#ifndef YAMPI_DETAIL_MPI_DELETE_HPP
# define YAMPI_DETAIL_MPI_DELETE_HPP

# include <type_traits>

# include <mpi.h>


namespace yampi
{
  namespace detail
  {
    template <typename T>
    struct mpi_delete
    {
      constexpr mpi_delete() noexcept = default;

      template <typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
      mpi_delete(mpi_delete<U> const&) noexcept
      { }

      void operator()(T* ptr) const
      {
        static_assert(not std::is_void<T>::value, "can't delete pointer to incomplete type");
        static_assert(sizeof(T) > 0, "can't delete pointer to incomplete type");
        ptr->~T();
        MPI_Free_mem(ptr);
      }
    };

    template <typename T>
    struct mpi_delete<T[]>
    {
      constexpr mpi_delete() noexcept = default;

      template <typename U, typename = typename std::enable_if<std::is_convertible<U(*)[], T(*)[]>::value>::type>
      mpi_delete(mpi_delete<U[]> const&) noexcept
      { }

      template <typename U>
      typename std::enable_if<std::is_convertible<U(*)[], T(*)[]>::value, void>::type
      operator()(U* ptr) const
      {
        static_assert(sizeof(T) > 0, "can't delete pointer to incomplete type");
        MPI_Free_mem(ptr);
      }
    };
  }
}


#endif

