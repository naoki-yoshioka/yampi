#ifndef YAMPI_DYNAMIC_WINDOW_ALLOCATOR_HPP
# define YAMPI_DYNAMIC_WINDOW_ALLOCATOR_HPP

# include <cassert>
# include <cstddef>
# include <cstdlib>
# include <limits>
# include <stdexcept>
# include <type_traits>
# include <memory>
# include <utility>

# include <mpi.h>

# include <yampi/is_initialized.hpp>
# include <yampi/is_finalized.hpp>
# include <yampi/dynamic_window.hpp>
# include <yampi/allocator.hpp> // not_yet_initialized_error, allocate_error, deallocate_error
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  class window_attach_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit window_attach_error(int const error_code)
     : std::runtime_error("Error occurred when attaching to window"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };

  class window_detach_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit window_detach_error(int const error_code)
     : std::runtime_error("Error occurred when detaching from window"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };

  class std_malloc_error
    : public std::runtime_error
  {
   public:
    std_malloc_error()
     : std::runtime_error("Error occurred when allocating using std::malloc")
    { }
  };


  namespace dynamic_window_allocator_detail
  {
    template <typename T, bool uses_speciall_memory>
    struct allocate;

    template <typename T>
    struct allocate<T, true>
    {
      typedef T* pointer;

      static pointer call(MPI_Win const& mpi_win, std::size_t const n)
      {
        assert(::yampi::is_initialized());

        pointer result;
        int const error_code1
          = MPI_Alloc_mem(static_cast<MPI_Aint>(n) * static_cast<MPI_Aint>(sizeof(T)), MPI_INFO_NULL, &result);
        if (error_code1 == MPI_SUCCESS)
          throw ::yampi::allocate_error(error_code1);

        int const error_code2
          = MPI_Win_attach(mpi_win, result, static_cast<MPI_Aint>(n) * static_cast<MPI_Aint>(sizeof(T)));
        return error_code2 == MPI_SUCCESS
          ? result
          : throw ::yampi::window_attach_error(error_code2);
      }
    };

    template <typename T>
    struct allocate<T, false>
    {
      typedef T* pointer;

      static pointer call(MPI_Win const& mpi_win, std::size_t const n)
      {
        assert(::yampi::is_initialized());

        pointer const result = reinterpret_cast<pointer>(std::malloc(n * sizeof(T)));
        if (result == nullptr)
          throw ::yampi::std_malloc_error();

        int const error_code
          = MPI_Win_attach(mpi_win, result, static_cast<MPI_Aint>(n) * static_cast<MPI_Aint>(sizeof(T)));
        return error_code == MPI_SUCCESS
          ? result
          : throw ::yampi::window_attach_error(error_code);
      }
    };

    template <typename T, bool uses_speciall_memory>
    struct deallocate;

    template <typename T>
    struct deallocate<T, true>
    {
      typedef T* pointer;

      static void call(MPI_Win const& mpi_win, pointer ptr)
      {
        int const error_code1 = MPI_Win_detach(mpi_win, ptr);
        if (error_code1 != MPI_SUCCESS)
          throw ::yampi::window_detach_error(error_code1);

        int const error_code2 = MPI_Free_mem(ptr);
        if (error_code2 != MPI_SUCCESS)
          throw ::yampi::deallocate_error(error_code2);
      }
    };

    template <typename T>
    struct deallocate<T, false>
    {
      typedef T* pointer;

      static void call(MPI_Win const& mpi_win, pointer ptr)
      {
        int const error_code = MPI_Win_detach(mpi_win, ptr);
        if (error_code != MPI_SUCCESS)
          throw ::yampi::window_detach_error(error_code);

        std::free(ptr);
      }
    };
  }


  template <typename T, bool uses_special_memory = true>
  class dynamic_window_allocator<T, uses_special_memory>
  {
    ::yampi::dynamic_window* window_ptr_;

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

    typedef std::false_type is_always_equal;

    template <typename U>
    struct rebind
    { typedef dynamic_window_allocator<U> other; };

    dynamic_window_allocator()
      : window_ptr_(nullptr)
    {
      if (not ::yampi::is_initialized())
        throw ::yampi::not_yet_initialized_error();

      if (::yampi::is_finalized())
        throw ::yampi::already_finalized_error(); // defined in environment.hpp
    }

    explicit dynamic_window_allocator(::yampi::dynamic_window& window)
      : window_ptr_(std::addressof(window))
    { }

    dynamic_window_allocator(dynamic_window_allocator const& other) noexcept
      : window_ptr_(other.window_ptr_)
    { }

    template <typename U>
    dynamic_window_allocator(dynamic_window_allocator<U, uses_special_memory> const& other) noexcept
      : window_ptr_(std::addressof(other.window()))
    { }

    bool operator==(dynamic_window_allocator const& other) const noexcept
    { return window_ptr_ == other.window_ptr_; }

    template <typename U>
    bool operator==(dynamic_window_allocator<U, uses_special_memory> const& other) const noexcept
    { return window_ptr_ == std::addressof(other.window()); }

    ::yampi::dynamic_window const& window() const { return *window_ptr_; }


    pointer address(reference x) const { return std::addressof(x); }
    const_pointer address(const_reference x) const { return std::addressof(x); }

    pointer allocate(std::size_t const n, const_void_pointer = nullptr)
    { return ::yampi::dynamic_window_allocator_detail::allocate<T, uses_special_memory>::call(window_ptr_->mpi_win(), n); }

    void deallocate(pointer ptr, std::size_t const)
    { return ::yampi::dynamic_window_allocator_detail::deallocate<T, uses_special_memory>::call(window_ptr_->mpi_win(), ptr); }

    size_type max_size() const noexcept
    { return std::numeric_limits<std::size_t>::max() / sizeof(T); }

    template <typename U, typename... Arguments>
    void construct(U* ptr, Arguments&&... arguments)
    { ::new((void *)ptr) U(std::forward<Arguments>(arguments)...); }

    void destroy(pointer ptr) { ((T*)ptr)->~T(); }
    template <typename U>
    void destroy(U* ptr) { ptr->~U(); }
  };

  template <bool uses_special_memory>
  class dynamic_window_allocator<void, uses_special_memory>
  {
   public:
    typedef void* pointer;
    typedef void const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef void value_type;

    typedef std::true_type is_always_equal;

    constexpr bool operator==(dynamic_window_allocator const&) const noexcept
    { return true; }

    template <typename T, uses_special_memory>
    constexpr bool operator==(dynamic_window_allocator<T, uses_special_memory> const&) const noexcept
    { return true; }

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };
  };

  template <typename T, typename U, bool uses_special_memory>
  inline bool operator!=(
    ::yampi::dynamic_window_allocator<T, uses_special_memory> const& lhs,
    ::yampi::dynamic_window_allocator<U, uses_special_memory> const& rhs) noexcept
  { return not (lhs == rhs); }
}
# endif // MPI_VERSION >= 3


#endif

