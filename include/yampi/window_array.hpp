#ifndef YAMPI_WINDOW_ARRAY_HPP
# define YAMPI_WINDOW_ARRAY_HPP

# include <cassert>
# include <cstddef>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/addressof.hpp>
# include <yampi/information.hpp>
# include <yampi/group.hpp>
# include <yampi/byte_displacement.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


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
              result, std::addressof(mpi_win));
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
              result, std::addressof(mpi_win));
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
      noexcept(std::is_nothrow_copy_constructible<MPI_Win>::value)
      : mpi_win_(MPI_WIN_NULL), base_ptr_(nullptr), num_elements_(std::size_t{0u})
    { }

    window_array(window_array const&) = delete;
    window_array& operator=(window_array const&) = delete;

    window_array(window_array&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
      : mpi_win_(std::move(other.mpi_win_)), base_ptr_(std::move(other.base_ptr_)), num_elements_(std::move(other.num_elements_))
    { other.mpi_win_ = MPI_WIN_NULL; other.base_ptr_ = nullptr; }

    window_array& operator=(window_array&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_win_ != MPI_WIN_NULL)
          MPI_Win_free(std::addressof(mpi_win_));
        mpi_win_ = std::move(other.mpi_win_);
        base_ptr_ = std::move(other.base_ptr_);
        num_elements_ = std::move(other.num_elements_);
        other.mpi_win_ = MPI_WIN_NULL;
        other.base_ptr_ = nullptr;
        other.num_elements_ = static_cast<std::size_t>(0u);
      }
      return *this;
    }

    ~window_array() noexcept
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      MPI_Win_free(std::addressof(mpi_win_));
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

    bool operator==(window_array const& other) const noexcept
    { return mpi_win_ == other.mpi_win_ and base_ptr_ == other.base_ptr_; }

    bool operator<(window_array const& other) const noexcept
    { return std::lexicographical_compare(begin(), end(), other.begin(), other.end()); }

    bool do_is_null() const noexcept
    { return mpi_win_ == MPI_WIN_NULL; }

    MPI_Win const& do_mpi_win() const noexcept
    { return mpi_win_; }

    template <typename U>
    U* do_base_ptr(::yampi::environment const& environment) const
    {
      U* base_ptr;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_BASE, base_ptr, std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? base_ptr
        : throw ::yampi::error(error_code, "yampi::window::do_base_ptr", environment);
    }

    ::yampi::byte_displacement do_size_bytes(::yampi::environment const& environment) const
    {
      MPI_Aint size_bytes;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_SIZE, std::addressof(size_bytes), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? ::yampi::byte_displacement(size_bytes)
        : throw ::yampi::error(error_code, "yampi::window::do_size_bytes", environment);
    }

    int do_displacement_unit(::yampi::environment const& environment) const
    {
      int displacement_unit;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_DISP_UNIT, std::addressof(displacement_unit), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? displacement_unit
        : throw ::yampi::error(error_code, "yampi::window::do_displacement_unit", environment);
    }

    ::yampi::flavor do_flavor(::yampi::environment const& environment) const
    {
      int flavor;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_CREATE_FLAVOR, std::addressof(flavor), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::flavor>(flavor)
        : throw ::yampi::error(error_code, "yampi::window::do_flavor", environment);
    }

    ::yampi::memory_model do_memory_model(::yampi::environment const& environment) const
    {
      int memory_model;
      int flag;
      int const error_code
        = MPI_Win_get_attr(mpi_win_, MPI_WIN_MODEL, std::addressof(memory_model), std::addressof(flag));
      return error_code == MPI_SUCCESS and flag
        ? static_cast<::yampi::memory_model>(memory_model)
        : throw ::yampi::error(error_code, "yampi::window::do_memory_model", environment);
    }

    void do_group(::yampi::group& group, ::yampi::environment const& environment) const
    {
      MPI_Group mpi_group;
      int const error_code = MPI_Win_get_group(mpi_win_, std::addressof(mpi_group));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array::do_group", environment);
      group.reset(mpi_group, environment);
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_win_ == MPI_WIN_NULL)
        return;

      int const error_code = MPI_Win_free(std::addressof(mpi_win_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::free", environment);
    }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(window_array&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_win_ = std::move(other.mpi_win_);
      base_ptr_ = std::move(other.base_ptr_);
      num_elements_ = std::move(other.num_elements_);
      other.mpi_win_ = MPI_WIN_NULL;
      other.base_ptr_ = nullptr;
      other.num_elements_ = static_cast<std::size_t>(0u);
    }

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
        throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::set_information", environment);
    }

    void get_information(yampi::information& information, yampi::environment const& environment) const
    {
      MPI_Info result;
      int const error_code = MPI_Win_get_info(mpi_win_, std::addressof(result));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::window_array<T, is_on_shared_memory>::get_information", environment);
      information.reset(result, environment);
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

    T* data() noexcept { return base_ptr_; }
    T const* data() const noexcept { return base_ptr_; }

    iterator begin() noexcept { return base_ptr_; }
    const_iterator begin() const noexcept { return base_ptr_; }
    const_iterator cbegin() const noexcept { return base_ptr_; }
    iterator end() noexcept { return base_ptr_ + num_elements_; }
    const_iterator end() const noexcept { return base_ptr_ + num_elements_; }
    const_iterator cend() const noexcept { return base_ptr_ + num_elements_; }
    reverse_iterator rbegin() noexcept { return reverse_iterator(this->end()); }
    const_reverse_iterator rbegin() const noexcept { return reverse_iterator(this->end()); }
    const_reverse_iterator crbegin() const noexcept { return reverse_iterator(this->cend()); }
    reverse_iterator rend() noexcept { return reverse_iterator(this->begin()); }
    const_reverse_iterator rend() const noexcept { return reverse_iterator(this->begin()); }
    const_reverse_iterator crend() const noexcept { return reverse_iterator(this->cbegin()); }

    bool empty() const noexcept { return num_elements_ == std::size_t{0u}; }
    size_type size() const  noexcept { return num_elements_; }
    size_type max_size() const  noexcept { return num_elements_; }

    void fill(T const& value) { std::fill(begin(), end(), value); }

    void swap(window_array& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Win>::value)
    {
      using std::swap;
      swap(mpi_win_, other.mpi_win_);
      swap(base_ptr_, other.base_ptr_);
      swap(num_elements_, other.num_elements_);
    }
  };

  template <typename T, bool is_on_shared_memory>
  inline bool operator!=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    noexcept
  { return not (lhs == rhs); }

  template <typename T, bool is_on_shared_memory>
  inline bool operator>(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    noexcept
  { return rhs < lhs; }

  template <typename T, bool is_on_shared_memory>
  inline bool operator<=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    noexcept
  { return not (lhs > rhs); }

  template <typename T, bool is_on_shared_memory>
  inline bool operator>=(
    ::yampi::window_array<T, is_on_shared_memory> const& lhs,
    ::yampi::window_array<T, is_on_shared_memory> const& rhs)
    noexcept
  { return not (lhs < rhs); }

  template <typename T, bool is_on_shared_memory>
  inline void swap(
    ::yampi::window_array<T, is_on_shared_memory>& lhs,
    ::yampi::window_array<T, is_on_shared_memory>& rhs)
    noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}
# endif // MPI_VERSION >= 3


# undef YAMPI_is_nothrow_swappable

#endif

