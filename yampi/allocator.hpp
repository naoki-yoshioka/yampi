#ifndef YAMPI_ALLOCATOR_HPP
# define YAMPI_ALLOCATOR_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <limits>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif
# if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#   include <utility>
# endif

# include <mpi.h>

# include <yampi/error.hpp>
# include <yampi/detail/workaround.hpp>
# include <yampi/is_initialized.hpp>
# include <yampi/environment.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_true_type std::true_type
# else
#   define YAMPI_true_type boost::true_type
# endif

# ifndef BOOST_NO_CXX11_NULLPTR
#   define YAMPI_nullptr nullptr
# else
#   define YAMPI_nullptr NULL
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename T>
  class allocator
  {
   public:
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
    using pointer = T*;
    using const_pointer = T const*;
    using void_pointer = void*;
    using const_void_pointer = void const*;
    using reference = T&;
    using const_reference = T const&;
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = YAMPI_true_type;
    using is_always_equal = YAMPI_true_type;

    template <typename U>
    struct rebind
    { using other = allocator<U>; };
# else
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    typedef YAMPI_true_type propagate_on_container_move_assignment;
    typedef YAMPI_true_type is_always_equal;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    BOOST_CONSTEXPR allocator() = default;
# else
    BOOST_CONSTEXPR allocator() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif

    allocator(allocator const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { }

    template <typename U>
    allocator(allocator<U> const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { }

    pointer address(reference x) const { return YAMPI_addressof(x); }
    const_pointer address(const_reference x) const { return YAMPI_addressof(x); }

    pointer allocate(std::size_t const n, const_void_pointer = YAMPI_nullptr)
    {
      assert(::yampi::is_initialized());

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = pointer{};
#   else
      auto result = pointer();
#   endif

      auto const error_code = MPI_Alloc_mem(static_cast<MPI_Aint>(n * sizeof(T)), MPI_INFO_NULL, &result);
# else
      pointer result;

      int const error_code = MPI_Alloc_mem(static_cast<MPI_Aint>(n * sizeof(T)), MPI_INFO_NULL, &result);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::allocator::allocate"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::allocator::allocate");
# endif

      ++::yampi::environment::num_unreleased_resources_;

      return result;
    }

    void deallocate(pointer ptr, std::size_t const)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code = MPI_Free_mem(ptr);
# else
      int const error_code = MPI_Free_mem(ptr);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::allocator::deallocate"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::allocator::deallocate");
# endif

      --::yampi::environment::num_unreleased_resources_;
      if (::yampi::environment::num_unreleased_resources_ == 0
          and not ::yampi::environment::is_initialized_)
      {
        if (not ::yampi::is_finalized())
          ::yampi::environment::error_code_on_last_finalize_ = MPI_Finalize();

        ::yampi::environment::is_initialized_ = false;
      }
    }

    size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW
    { return std::numeric_limits<std::size_t>::max() / sizeof(T); }

# if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
    template <typename U, typename... Arguments>
    void construct(U* ptr, Arguments&&... arguments)
    { ::new((void *)ptr) U(YAMPI_DETAIL_forward<Arguments>(arguments)...); }
# else
    void construct(pointer ptr, const_reference value)
    { new((void *)ptr) T(value); }
# endif

    void destroy(pointer ptr) { ((T*)ptr)->~T(); }
    template <typename U>
    void destroy(U* ptr) { ptr->~U(); }
  };

  template <>
  class allocator<void>
  {
   public:
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
    using pointer = void*;
    using const_pointer = void const*;
    using void_pointer = void*;
    using const_void_pointer = void const*;
    using value_type = void;

    template <typename U>
    struct rebind
    { using other = allocator<U>; };
# else
    typedef void* pointer;
    typedef void const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef void value_type;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };
# endif
  };


  template <typename T, typename U>
  inline BOOST_CONSTEXPR bool operator==(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) BOOST_NOEXCEPT_OR_NOTHROW
  { return true; }

  template <typename T, typename U>
  inline BOOST_CONSTEXPR bool operator!=(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) BOOST_NOEXCEPT_OR_NOTHROW
  { return false; }
}


# undef YAMPI_true_type
# undef YAMPI_nullptr
# undef YAMPI_addressof

#endif

