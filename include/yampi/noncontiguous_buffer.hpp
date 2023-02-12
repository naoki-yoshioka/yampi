#ifndef YAMPI_NONCONTIGUOUS_BUFFER_HPP
# define YAMPI_NONCONTIGUOUS_BUFFER_HPP

# include <cassert>
# include <iterator>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <boost/range/value_type.hpp>

# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/has_predefined_datatype.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  template <typename T, bool enables_multiple_datatypes = false, typename Enable = void>
  class noncontiguous_buffer
  {
    T* data_;
    int* count_first_;
    int* displacement_first_;
    ::yampi::datatype* datatype_ptr_;

   public:
    template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
    noncontiguous_buffer(
      ContiguousIterator1 const first, ContiguousIterator2 const count_first,
      ContiguousIterator3 const displacement_first,
      ::yampi::datatype const& datatype)
      : data_{std::addressof(*first)},
        count_first_{std::addressof(*count_first)},
        displacement_first_{std::addressof(*displacement_first)},
        datatype_ptr_{const_cast< ::yampi::datatype* >(std::addressof(datatype))}
    {
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator1>::value_type, T>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator2>::value_type, int>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator3>::value_type, int>::value);
    }

    bool operator==(noncontiguous_buffer const& other) const noexcept
    {
      return data_ == other.data_
        and count_first_ == other.count_first_
        and displacement_first_ == other.displacement_first_
        and *datatype_ptr_ == *other.datatype_ptr_;
    }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }

    int* count_first() noexcept { return count_first_; }
    int const* count_first() const noexcept { return count_first_; }

    int* displacement_first() noexcept { return displacement_first_; }
    int const* displacement_first() const noexcept { return displacement_first_; }

    ::yampi::datatype const& datatype() const noexcept { return *datatype_ptr_; }

    void swap(noncontiguous_buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_first_, other.count_first_);
      swap(displacement_first_, other.displacement_first_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class noncontiguous_buffer<T, enables_multiple_datatypes, Enable>

  template <typename T>
  class noncontiguous_buffer<T, true, typename std::enable_if<not ::yampi::has_predefined_datatype<T>::value>::type>
  {
    T* data_;
    int* count_first_;
    int* displacement_first_;
    ::yampi::datatype* datatype_first_;

   public:
    template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3, typename ContiguousIterator4>
    noncontiguous_buffer(
      ContiguousIterator1 const first, ContiguousIterator2 const count_first,
      ContiguousIterator3 const displacement_first, ContiguousIterator4 const datatype_first)
      : data_{std::addressof(*first)},
        count_first_{std::addressof(*count_first)},
        displacement_first_{std::addressof(*displacement_first)},
        datatype_first_{std::addressof(*datatype_first)}
    {
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator1>::value_type, T>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator2>::value_type, int>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator3>::value_type, int>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator4>::value_type, ::yampi::datatype>::value);
    }

    bool operator==(noncontiguous_buffer const& other) const noexcept
    {
      return data_ == other.data_
        and count_first_ == other.count_first_
        and displacement_first_ == other.displacement_first_
        and datatype_first_ == other.datatype_first_;
    }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }

    int* count_first() noexcept { return count_first_; }
    int const* count_first() const noexcept { return count_first_; }

    int* displacement_first() noexcept { return displacement_first_; }
    int const* displacement_first() const noexcept { return displacement_first_; }

    ::yampi::datatype* datatype_first() noexcept { return datatype_first_; }
    ::yampi::datatype const* datatype_first() const noexcept { return datatype_first_; }
    // returns first datatype
    ::yampi::datatype const& datatype() const noexcept { return *datatype_first_; }

    void swap(noncontiguous_buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_first_, other.count_first_);
      swap(displacement_first_, other.displacement_first_);
      swap(datatype_first_, other.datatype_first_);
    }
  }; // class noncontiguous_buffer<T, true, typename std::enable_if<not ::yampi::has_predefined_datatype<T>::value>::type>

  template <typename T>
  class noncontiguous_buffer<T, false, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    T* data_;
    int* count_first_;
    int* displacement_first_;

   public:
    template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
    noncontiguous_buffer(
      ContiguousIterator1 const first, ContiguousIterator2 const count_first,
      ContiguousIterator3 const displacement_first)
      : data_{std::addressof(*first)},
        count_first_{std::addressof(*count_first)},
        displacement_first_{std::addressof(*displacement_first)}
    {
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator1>::value_type, T>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator2>::value_type, int>::value);
      static_assert(std::is_convertible<typename std::iterator_traits<ContiguousIterator3>::value_type, int>::value);
    }

    bool operator==(noncontiguous_buffer const& other) const noexcept
    {
      return data_ == other.data_
        and count_first_ == other.count_first_
        and displacement_first_ == other.displacement_first_;
    }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }

    int* count_first() noexcept { return count_first_; }
    int const* count_first() const noexcept { return count_first_; }

    int* displacement_first() noexcept { return displacement_first_; }
    int const* displacement_first() const noexcept { return displacement_first_; }

    ::yampi::predefined_datatype<T> datatype() const noexcept { return ::yampi::predefined_datatype<T>(); }

    void swap(noncontiguous_buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_first_, other.count_first_);
      swap(displacement_first_, other.displacement_first_);
    }
  }; // class noncontiguous_buffer<T, false, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  class noncontiguous_buffer<T, true, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
    = delete;

  template <typename T, bool enables_multiple_datatypes>
  inline bool operator!=(
    ::yampi::noncontiguous_buffer<T, enables_multiple_datatypes> const& lhs,
    ::yampi::noncontiguous_buffer<T, enables_multiple_datatypes> const& rhs)
    noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T, bool enables_multiple_datatypes>
  inline void swap(
    ::yampi::noncontiguous_buffer<T, enables_multiple_datatypes>& lhs,
    ::yampi::noncontiguous_buffer<T, enables_multiple_datatypes>& rhs)
   noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator1>::value_type>::type>::value,
    ::yampi::noncontiguous_buffer<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator1>::value_type>::type, false>>::type
  make_noncontiguous_buffer(
    ContiguousIterator1 const first, ContiguousIterator2 const count_first,
    ContiguousIterator3 const displacement_first)
  { return {first, count_first, displacement_first}; }

  template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
  inline ::yampi::noncontiguous_buffer<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator1>::value_type>::type, false>
  make_noncontiguous_buffer(
    ContiguousIterator1 const first, ContiguousIterator2 const count_first,
    ContiguousIterator3 const displacement_first,
    ::yampi::predefined_datatype<typename std::remove_cv<typename std::iterator_traits<ContiguouIterator1>::value_type>::type> const&)
  { return {first, count_first, displacement_first}; }

  template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
  inline ::yampi::noncontiguous_buffer<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator1>::value_type>::type, false>
  make_noncontiguous_buffer(
    ContiguousIterator1 const first, ContiguousIterator2 const count_first,
    ContiguousIterator3 const displacement_first, ::yampi::datatype const& datatype)
  { return {first, count_first, displacement_first, datatype}; }

  template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3, typename ContiguousIterator4>
  inline ::yampi::noncontiguous_buffer<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator1>::value_type>::type, true>
  make_noncontiguous_buffer(
    ContiguousIterator1 const first, ContiguousIterator2 const count_first,
    ContiguousIterator3 const displacement_first, ContiguousIterator4 const datatype_first)
  { return {first, count_first, displacement_first, datatype_first}; }
}


# undef YAMPI_is_nothrow_swappable

#endif
