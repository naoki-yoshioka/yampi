#ifndef YAMPI_BUFFER_HPP
# define YAMPI_BUFFER_HPP

# include <cassert>
# include <iterator>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
#   include <memory>

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
  template <typename T, typename Enable = void>
  class buffer
  {
    T* data_;
    int count_;
    ::yampi::datatype const* datatype_ptr_;

   public:
    buffer(T& value, ::yampi::datatype const& datatype) noexcept
      : data_{std::addressof(value)}, count_{1},
        datatype_ptr_{std::addressof(datatype)}
    { }

    buffer(T const& value, ::yampi::datatype const& datatype) noexcept
      : data_{const_cast<T*>(std::addressof(value))}, count_{1},
        datatype_ptr_{std::addressof(datatype)}
    { }

    template <typename ContiguousIterator>
    buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype const& datatype)
      noexcept(noexcept(*first) and noexcept(last-first))
      : data_{const_cast<T*>(std::addressof(*first))},
        count_{last-first},
        datatype_ptr_{std::addressof(datatype)}
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           T>::value),
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    bool operator==(buffer const& other) const noexcept
    { return data_ == other.data_ and count_ == other.count_ and *datatype_ptr_ == *other.datatype_ptr_; }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }
    int const& count() const noexcept { return count_; }
    ::yampi::datatype const& datatype() const noexcept { return *datatype_ptr_; }

    void swap(buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_, other.count_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class buffer<T, Enable>

  template <typename T>
  class buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    T* data_;
    int count_;

   public:
    explicit buffer(T& value) noexcept
      : data_{std::addressof(value)}, count_{1}
    { }

    explicit buffer(T const& value) noexcept
      : data_{const_cast<T*>(std::addressof(value))}, count_{1}
    { }

    template <typename ContiguousIterator>
    buffer(ContiguousIterator const first, ContiguousIterator const last)
      noexcept(noexcept(*first) and noexcept(last-first))
      : data_{const_cast<T*>(std::addressof(*first))},
        count_{static_cast<int>(last-first)}
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           T>::value),
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    bool operator==(buffer const& other) const noexcept
    { return data_ == other.data_ and count_ == other.count_; }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }
    int const& count() const noexcept { return count_; }
    ::yampi::predefined_datatype<T> datatype() const noexcept { return ::yampi::predefined_datatype<T>(); }

    void swap(buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_, other.count_);
    }
  }; // class buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  inline bool operator!=(::yampi::buffer<T> const& lhs, ::yampi::buffer<T> const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::buffer<T>& lhs, ::yampi::buffer<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename T>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::buffer<T> >::type make_buffer(T& value)
    noexcept(noexcept(::yampi::buffer<T>(value)))
  { return ::yampi::buffer<T>(value); }

  template <typename T>
  inline
  typename std::enable_if< ::yampi::has_predefined_datatype<T>::value, ::yampi::buffer<T> >::type make_buffer(T const& value)
    noexcept(noexcept(::yampi::buffer<T>(value)))
  { return ::yampi::buffer<T>(value); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T& value, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::buffer<T>(value)))
  { return ::yampi::buffer<T>(value); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T const& value, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::buffer<T>(value)))
  { return ::yampi::buffer<T>(value); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T& value, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::buffer<T>(value, datatype)))
  { return ::yampi::buffer<T>(value, datatype); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T const& value, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::buffer<T>(value, datatype)))
  { return ::yampi::buffer<T>(value, datatype); }

  template <typename ContiguousIterator>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<
      typename std::remove_cv<
        typename std::iterator_traits<ContiguousIterator>::value_type>::type>::value,
    ::yampi::buffer<
      typename std::remove_cv<
        typename std::iterator_traits<ContiguousIterator>::value_type>::type>
  >::type make_buffer(ContiguousIterator const first, ContiguousIterator const last)
    noexcept(
      noexcept(
        ::yampi::buffer<
          typename std::remove_cv<
            typename std::iterator_traits<ContiguousIterator>::value_type>::type>(
          first, last)))
  {
    typedef
      ::yampi::buffer<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type>
      result_type;
    return result_type(first, last);
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename std::iterator_traits<ContiguousIterator>::value_type>::type>
  make_buffer(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::predefined_datatype<typename std::remove_cv<typename std::iterator_traits<ContiguousIterator>::value_type>::type> const&)
    noexcept(
      noexcept(
        ::yampi::buffer<
          typename std::remove_cv<
            typename std::iterator_traits<ContiguousIterator>::value_type>::type>(
          first, last)))
  {
    typedef
      ::yampi::buffer<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type>
      result_type;
    return result_type(first, last);
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename std::iterator_traits<ContiguousIterator>::value_type>::type>
  make_buffer(ContiguousIterator const first, ContiguousIterator const last, ::yampi::datatype const& datatype)
    noexcept(
      noexcept(
        ::yampi::buffer<
          typename std::remove_cv<
            typename std::iterator_traits<ContiguousIterator>::value_type>::type>(
          first, last, datatype)))
  {
    typedef
      ::yampi::buffer<
        typename std::remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type>
      result_type;
    return result_type(first, last, datatype);
  }

  template <typename ContiguousRange>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<
      typename std::remove_cv<
        typename boost::range_value<ContiguousRange>::type>::type>::value,
    ::yampi::buffer<
      typename std::remove_cv<
        typename boost::range_value<ContiguousRange>::type>::type>
  >::type range_to_buffer(ContiguousRange& range)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<
      typename std::remove_cv<
        typename boost::range_value<ContiguousRange const>::type>::type>::value,
    ::yampi::buffer<
      typename std::remove_cv<
        typename boost::range_value<ContiguousRange const>::type>::type>
  >::type range_to_buffer(ContiguousRange const& range)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename boost::range_value<ContiguousRange>::type>::type>
  range_to_buffer(
    ContiguousRange& range,
    ::yampi::predefined_datatype<typename std::remove_cv<typename boost::range_value<ContiguousRange>::type>::type> const&)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename boost::range_value<ContiguousRange const>::type>::type>
  range_to_buffer(
    ContiguousRange const& range,
    ::yampi::predefined_datatype<typename std::remove_cv<typename boost::range_value<ContiguousRange const>::type>::type> const&)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename boost::range_value<ContiguousRange>::type>::type>
  range_to_buffer(ContiguousRange& range, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range), datatype)))
  { return ::yampi::make_buffer(std::begin(range), std::end(range), datatype); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename std::remove_cv<
      typename boost::range_value<ContiguousRange const>::type>::type>
  range_to_buffer(ContiguousRange const& range, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range), datatype)))
  { return ::yampi::make_buffer(std::begin(range), std::end(range), datatype); }
}


# undef YAMPI_is_nothrow_swappable

#endif
