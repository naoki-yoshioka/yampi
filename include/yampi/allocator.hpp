#ifndef YAMPI_ALLOCATOR_HPP
# define YAMPI_ALLOCATOR_HPP

# include <cassert>
# include <cstddef>
# include <limits>
# include <stdexcept>
# include <type_traits>
# include <memory>
# include <utility>

# include <mpi.h>

# include <yampi/is_initialized.hpp>
# include <yampi/is_finalized.hpp>
# include <yampi/environment.hpp>


namespace yampi
{
  class not_yet_initialized_error
    : public std::logic_error
  {
   public:
    not_yet_initialized_error()
      : std::logic_error{"MPI environment has not been initialized yet"}
    { }
  };


  class allocate_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit allocate_error(int const error_code)
     : std::runtime_error{"Error occurred when allocating"},
       error_code_{error_code}
    { }

    int error_code() const { return error_code_; }
  };

  class deallocate_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit deallocate_error(int const error_code)
     : std::runtime_error{"Error occurred when deallocating"},
       error_code_{error_code}
    { }

    int error_code() const { return error_code_; }
  };


  template <typename T>
  class allocator
  {
   public:
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::true_type is_always_equal;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };

    allocator()
    {
      if (not ::yampi::is_initialized())
        throw ::yampi::not_yet_initialized_error();

      if (::yampi::is_finalized())
        throw ::yampi::already_finalized_error(); // defined in environment.hpp
    }

    explicit allocator(::yampi::environment const&) noexcept { }

    allocator(allocator const& other) noexcept { }

    template <typename U>
    allocator(allocator<U> const& other) noexcept { }

    pointer address(reference x) const { return std::addressof(x); }
    const_pointer address(const_reference x) const { return std::addressof(x); }

    pointer allocate(std::size_t const n, const_void_pointer = nullptr)
    {
      assert(::yampi::is_initialized());

      pointer result;
      int const error_code
        = MPI_Alloc_mem(static_cast<MPI_Aint>(n) * static_cast<MPI_Aint>(sizeof(T)), MPI_INFO_NULL, &result);
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::allocate_error(error_code);
    }

    void deallocate(pointer ptr, std::size_t const)
    {
      int const error_code = MPI_Free_mem(ptr);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::deallocate_error(error_code);
    }

    size_type max_size() const noexcept
    { return std::numeric_limits<std::size_t>::max() / sizeof(T); }

    template <typename U, typename... Arguments>
    void construct(U* ptr, Arguments&&... arguments)
    { ::new((void *)ptr) U(std::forward<Arguments>(arguments)...); }

    void destroy(pointer ptr) { ((T*)ptr)->~T(); }
    template <typename U>
    void destroy(U* ptr) { ptr->~U(); }
  };

  template <>
  class allocator<void>
  {
   public:
    typedef void* pointer;
    typedef void const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef void value_type;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };
  };


  template <typename T, typename U>
  inline constexpr bool operator==(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) noexcept
  { return true; }

  template <typename T, typename U>
  inline constexpr bool operator!=(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) noexcept
  { return false; }
}


#endif

