#ifndef YAMPI_WINDOW_ARRAY_HPP
# define YAMPI_WINDOW_ARRAY_HPP

# include <cassert>
# include <cstddef>
# include <algorithm>
# include <iterator>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/window_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/information.hpp>


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
# if MPI_VERSION >= 4
        int const error_code
          = MPI_Win_allocate_c(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<MPI_Aint>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, std::addressof(mpi_win));
# else // MPI_VERSION >= 4
        int const error_code
          = MPI_Win_allocate(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<int>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, std::addressof(mpi_win));
# endif // MPI_VERSION >= 4
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
# if MPI_VERSION >= 4
        int const error_code
          = MPI_Win_allocate_shared_c(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<MPI_Aint>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, std::addressof(mpi_win));
# else // MPI_VERSION >= 4
        int const error_code
          = MPI_Win_allocate_shared(
              static_cast<MPI_Aint>(sizeof(T)) * static_cast<MPI_Aint>(num_elements),
              static_cast<int>(sizeof(T)), mpi_info, communicator.mpi_comm(),
              result, std::addressof(mpi_win));
# endif // MPI_VERSION >= 4
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
    typedef ::yampi::window_base< ::yampi::window_array<T, is_on_shared_memory> > base_type;

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

    window_array() noexcept(noexcept(base_type{}))
      : base_type{}, base_ptr_{nullptr}, num_elements_{std::size_t{0u}}
    { }

    window_array(window_array const&) = delete;
    window_array& operator=(window_array const&) = delete;

    window_array(window_array&& other)
      noexcept(noexcept(base_type{std::move(other)}))
      : base_ptr_{std::move(other.base_ptr_)}, num_elements_{std::move(other.num_elements_)}, base_type{std::move(other)}
    { other.base_ptr_ = nullptr; }

    window_array& operator=(window_array&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Win>::value
        and std::is_nothrow_copy_assignable<MPI_Win>::value)
    {
      if (this != std::addressof(other))
      {
        base_ptr_ = std::move(other.base_ptr_);
        num_elements_ = std::move(other.num_elements_);
        other.base_ptr_ = nullptr;
        other.num_elements_ = static_cast<std::size_t>(0u);
      }
      base_type::operator=(std::move(other));
      return *this;
    }

    ~window_array() noexcept = default;

    window_array(
      std::size_t const num_elements,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{},
        base_ptr_{
          ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, MPI_INFO_NULL, communicator, environment)},
        num_elements_{num_elements}
    { }

    window_array(
      std::size_t const num_elements, ::yampi::information const& information,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{},
        base_ptr_{
          ::yampi::window_array_detail::create<T, is_on_shared_memory>::call(
            mpi_win_, num_elements, information.mpi_info(), communicator, environment)},
        num_elements_{num_elements}
    { }

    bool operator<(window_array const& other) const noexcept
    { return std::lexicographical_compare(begin(), end(), other.begin(), other.end()); }

   private:
    friend base_type;

    void do_reset(window_array&& other, ::yampi::environment const& environment)
    {
      base_ptr_ = std::move(other.base_ptr_);
      num_elements_ = std::move(other.num_elements_);
      other.base_ptr_ = nullptr;
      other.num_elements_ = static_cast<std::size_t>(0u);
    }

    void do_swap(window_array& other) noexcept
    {
      using std::swap;
      swap(base_ptr_, other.base_ptr_);
      swap(num_elements_, other.num_elements_);
    }

   public:
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
  };

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
}
# endif // MPI_VERSION >= 3


#endif

