#ifndef YAMPI_WINDOW_ARRAY_HPP
# define YAMPI_WINDOW_ARRAY_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <algorithm>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>
# include <yampi/group.hpp>
# include <yampi/byte_displacement.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_SCOPED_ENUMS
#     define YAMPI_FLAVOR ::yampi::flavor
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model
#   else // BOOST_NO_CXX11_SCOPED_ENUMS
#     define YAMPI_FLAVOR ::yampi::flavor::flavor_
#     define YAMPI_MEMORY_MODEL ::yampi::memory_model::memory_model_
#   endif // BOOST_NO_CXX11_SCOPED_ENUMS
# endif // MPI_VERSION >= 3


# if MPI_VERSION >= 3
namespace yampi
{
  namespace window_array_detail
  {
    template <typename T, bool is_on_shared_memory>
    struct create;

    template <typename T>
    struct create<T, false>
    {
      static T* call(
        MPI_Win& mpi_win, std::size_t const num_elements, MPI_Info const& mpi_info,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      {
        T* result;
        int const error_code
          = MPI_Win_allocate(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<int>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, YAMPI_addressof(mpi_win));
        return error_code == MPI_SUCCESS
          ? result
          : throw ::yampi::error(error_code, "yampi::window_array_detail::create<T, false>::call", environment);
      }
    };

    template <typename T>
    struct create<T, true>
    {
      static T* call(
        MPI_Win& mpi_win, std::size_t const num_elements, MPI_Info const& mpi_info,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      {
        T* result;
        int const error_code
          = MPI_Win_allocate_shared(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<int>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, YAMPI_addressof(mpi_win));
        return error_code == MPI_SUCCESS
          ? result
          : throw ::yampi::error(error_code, "yampi::window_array_detail::create<T, true>::call", environment);
      }
    };
  }

  template <typename T, bool is_on_shared_memory = false>
  class window_array
    : public ::yampi::window_base< ::yampi::window_array<T, is_on_shared_memory> >
  {
    typedef ::yampi::window_base< ::yampi::window_array<T, is_on_shared_memory> > super_type;

    MPI_Win mpi_win_;
    T* base_ptr_;
    std::size_t num_elements_;

   public:
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> reverse_const_iterator;

    window_array()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_(MPI_WIN_NULL), base_ptr_(nullptr), num_elements_(std::size_t{0u})
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    window_array(window_array const&) = delete;
    window_array& operator=(window_array const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    window_array(window_array const&);
    window_array& operator=(window_array const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    window_array(window_array&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_(std::move(other.mpi_win_)), base_ptr_(std::move(other.base_ptr_)), num_elements_(std::move(other.num_elements_))
    { other.mpi_win_ = MPI_WIN_NULL; other.base_ptr_ = nullptr; }

    window_array& operator=(window_array&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Win>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_win_ = std::move(other.mpi_win_);
        base_ptr_ = std::move(other.base_ptr_);
        num_elements_ = std::move(other.num_elements_);
        other.mpi_win_ = MPI_WIN_NULL;
        other.base_ptr_ = nullptr;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~window_array() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(YAMPI_addressof(mpi_win_));
    }

    window_array(
      std::size_t const num_elements,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : mpi_win_(),
        base_ptr_(
          ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, MPI_INFO_NULL, communicator, environment)),
        num_elements_(num_elements)
    { }

    window_array(
      std::size_t const num_elements, ::yampi::information const& information,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : mpi_win_(),
        base_ptr_(
          ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, information.mpi_info(), communicator, environment)),
        num_elements_(num_elements)
    { }

    bool operator==(window_array const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_ == other.mpi_win_ and base_ptr_ == other.base_ptr_; }

    bool operator<(window_array const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return std::lexicographical_compare(begin(), end(), other.begin(), other.end()); }

    bool do_is_null() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_ == MPI_WIN_NULL; }

    MPI_Win const& do_mpi_win() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_win_; }

    template <typename U>
    U* do_base_ptr() const
    {
      U* base_ptr;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_BASE, base_ptr, YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? base_ptr
        : throw ::yampi::error(error_code, "yampi::window::do_base_ptr", environment);
    }

    ::yampi::byte_displacement do_size_in_bytes() const
    {
      MPI_Aint size_in_bytes;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_SIZE, YAMPI_addressof(size_in_bytes), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? ::yampi::byte_displacement(size_in_bytes)
        : throw ::yampi::error(error_code, "yampi::window::do_size_in_bytes", environment);

    int do_displacement_unit() const
    {
      int displacement_unit;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_DISP_UNIT, YAMPI_addressof(displacement_unit), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? displacement_unit
        : throw ::yampi::error(error_code, "yampi::window::do_displacement_unit", environment);
    }

    YAMPI_FLAVOR do_flavor() const
    {
      int flavor;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_CREATE_FLAVOR, YAMPI_addressof(flavor), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<YAMPI_FLAVOR>(flavor)
        : throw ::yampi::error(error_code, "yampi::window::do_flavor", environment);
    }

    YAMPI_MEMORY_MODEL do_memory_model() const
    {
      int memory_model;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_MODEL, YAMPI_addressof(memory_model), YAMPI_addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<YAMPI_MEMORY_MODEL>(memory_model)
        : throw ::yampi::error(error_code, "yampi::window::do_memory_model", environment);
    }

    void do_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Win_get_group(mpi_win_, YAMPI_addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array::do_group", environment);
      group.reset(mpi_group, environment);
    }


    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(YAMPI_addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::free", environment);
    }


    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(
      std::size_t const num_elements,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      base_ptr_
        = ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, MPI_INFO_NULL, communicator, environment);
    }

    void reset(
      std::size_t const num_elements, ::yampi::information const& information,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      base_ptr_
        = ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, information.mpi_info(), communicator, environment);
    }

    void set_information(yampi::information const& information, yampi::environment const& environment) const
    {
      int const error_code = MPI_Win_set_info(mpi_win_, information.mpi_info());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::information", environment);
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? yampi::information(result)
        : throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::information", environment);
    }


    reference at(size_type const index)
    {
      return index < num_elements_
        ? base_ptr_[index]
        : throw std::out_of_range("yampi::window_array<T, is_on_shared_memory>::at");
    }

    const_reference at(size_type const index) const
    {
      return index < num_elements_
        ? base_ptr_[index]
        : throw std::out_of_range("yampi::window_array<T, is_on_shared_memory>::at");
    }

    reference operator[](size_type const index)
    { assert(index < num_elements_); return base_ptr_[index]; }

    const_reference operator[](size_type const index) const
    { assert(index < num_elements_); return base_ptr_[index]; }

    reference front() { return *base_ptr_; }
    const_reference front() const { return *base_ptr_; }

    reference back() { return base_ptr_[num_elements_-1]; }
    const_reference back() const { return base_ptr_[num_elements_-1]; }

    T* data() BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_; }
    T const* data() const BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_; }

    iterator begin() BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_; }
    const_iterator begin() const BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_; }
    const_iterator cbegin() const BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_; }
    iterator end() BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_ + num_elements_; }
    const_iterator end() const BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_ + num_elements_; }
    const_iterator cend() const BOOST_NOEXCEPT_OR_NOTHROW { return base_ptr_ + num_elements_; }
    reverse_iterator rbegin() BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->end()); }
    const_reverse_iterator rbegin() const BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->end()); }
    const_reverse_iterator crbegin() const BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->cend()); }
    reverse_iterator rend() BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->begin()); }
    const_reverse_iterator rend() const BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->begin()); }
    const_reverse_iterator crend() const BOOST_NOEXCEPT_OR_NOTHROW { return reverse_iterator(this->cbegin()); }

    bool empty() const BOOST_NOEXCEPT_OR_NOTHROW { return num_elements_ == std::size_t{0u}; }
    size_type size() const  BOOST_NOEXCEPT_OR_NOTHROW { return num_elements_; }
    size_type max_size() const  BOOST_NOEXCEPT_OR_NOTHROW { return num_elements_; }

    void fill(T const& value) { std::fill(begin(), end(), value); }

    void swap(window_array& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
      swap(base_ptr_, other.base_ptr_);
    }
  };

  template <typename T, bool is_on_shared_memory>
  inline bool operator!=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  template <typename T, bool is_on_shared_memory>
  inline bool operator>(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  template <typename T, bool is_on_shared_memory>
  inline bool operator<=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs > rhs); }

  template <typename T, bool is_on_shared_memory>
  inline bool operator>=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs < rhs); }

  template <typename T, bool is_on_shared_memory>
  inline void swap(
    ::yampi::window_array<T, is_on_shared_memory>& lhs,
    ::yampi::window_array<T, is_on_shared_memory>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}
# endif // MPI_VERSION >= 3


# if MPI_VERSION >= 3
#   undef YAMPI_MEMORY_MODEL
#   undef YAMPI_FLAVOR
# endif // MPI_VERSION >= 3
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_remove_cv

#endif

