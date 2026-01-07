#ifndef YAMPI_BUFFER_HPP
# define YAMPI_BUFFER_HPP

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
# if MPI_VERSION >= 4
#   include <yampi/count.hpp>
# endif

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
   public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using iterator = T*;
    using const_iterator = T const*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

   private:
    value_type* data_;
# if MPI_VERSION >= 4
    ::yampi::count count_;
# else // MPI_VERSION >= 4
    int count_;
# endif
    ::yampi::datatype* datatype_ptr_;

   public:
    buffer(T& value, ::yampi::datatype const& datatype) noexcept
      : data_{const_cast<value_type*>(std::addressof(value))}, count_{1},
        datatype_ptr_{const_cast< ::yampi::datatype* >(std::addressof(datatype))}
    { }

# if MPI_VERSION >= 4
    template <typename ContiguousIterator>
    buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype const& datatype)
      noexcept(noexcept(*first) and noexcept(last-first))
      : data_{const_cast<value_type*>(std::addressof(*first))},
        count_{static_cast<MPI_Count>(last-first)},
        datatype_ptr_{const_cast< ::yampi::datatype* >(std::addressof(datatype))}
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           value_type>::value),
        "CV-removed T should be tha same to cv-removed value_type of ContiguousIterator");
      assert(last >= first);
    }
# else // MPI_VERSION >= 4
    template <typename ContiguousIterator>
    buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype const& datatype)
      noexcept(noexcept(*first) and noexcept(last-first))
      : data_{const_cast<T*>(std::addressof(*first))},
        count_{static_cast<int>(last-first)},
        datatype_ptr_{const_cast< ::yampi::datatype* >(std::addressof(datatype))}
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           value_type>::value),
        "CV-removed T should be tha same to cv-removed value_type of ContiguousIterator");
      assert(last >= first);
    }
# endif // MPI_VERSION >= 4

    bool operator==(buffer const& other) const noexcept
    { return data_ == other.data_ and count_ == other.count_ and *datatype_ptr_ == *other.datatype_ptr_; }

    pointer data() noexcept { return data_; }
    const_pointer data() const noexcept { return data_; }
# if MPI_VERSION >= 4
    ::yampi::count const& count() const noexcept { return count_; }
# else // MPI_VERSION >= 4
    int const& count() const noexcept { return count_; }
# endif
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
   public:
    using element_type = T;
    using value_type = typename std::remove_cv<T>::type;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using iterator = T*;
    using const_iterator = T const*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

   private:
    value_type* data_;
# if MPI_VERSION >= 4
    ::yampi::count count_;
# else // MPI_VERSION >= 4
    int count_;
# endif

   public:
    explicit buffer(T& value) noexcept
      : data_{const_cast<value_type*>(std::addressof(value))}, count_{1}
    { }

# if MPI_VERSION >= 4
    template <typename ContiguousIterator>
    buffer(ContiguousIterator const first, ContiguousIterator const last)
      noexcept(noexcept(*first) and noexcept(last-first))
      : data_{const_cast<T*>(std::addressof(*first))},
        count_{static_cast<MPI_Count>(last-first)}
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           value_type>::value),
        "CV-removed T should be tha same to cv-removed value_type of ContiguousIterator");
      assert(last >= first);
    }
# else // MPI_VERSION >= 4
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
           value_type>::value),
        "CV-removed T should be tha same to cv-removed value_type of ContiguousIterator");
      assert(last >= first);
    }
# endif // MPI_VERSION >= 4

    bool operator==(buffer const& other) const noexcept
    { return data_ == other.data_ and count_ == other.count_; }

    pointer data() noexcept { return data_; }
    const_pointer data() const noexcept { return data_; }
# if MPI_VERSION >= 4
    ::yampi::count const& count() const noexcept { return count_; }
# else // MPI_VERSION >= 4
    int const& count() const noexcept { return count_; }
# endif
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
  inline ::yampi::buffer<T> make_buffer(T& value, ::yampi::predefined_datatype<T> const&)
    noexcept(noexcept(::yampi::buffer<T>(value)))
  { return ::yampi::buffer<T>(value); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T& value, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::buffer<T>(value, datatype)))
  { return ::yampi::buffer<T>(value, datatype); }

  template <typename ContiguousIterator>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
  >::type make_buffer(ContiguousIterator const first, ContiguousIterator const last)
    noexcept(
      noexcept(
        ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>(
          first, last)))
  {
    typedef
      ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
      result_type;
    return result_type(first, last);
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
  make_buffer(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::predefined_datatype<typename std::iterator_traits<ContiguousIterator>::value_type> const&)
    noexcept(
      noexcept(
        ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>(
          first, last)))
  {
    typedef
      ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
      result_type;
    return result_type(first, last);
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
  make_buffer(ContiguousIterator const first, ContiguousIterator const last, ::yampi::datatype const& datatype)
    noexcept(
      noexcept(
        ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>(
          first, last, datatype)))
  {
    typedef
      ::yampi::buffer<typename std::iterator_traits<ContiguousIterator>::value_type>
      result_type;
    return result_type(first, last, datatype);
  }

  template <typename ContiguousRange>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<typename boost::range_value<ContiguousRange>::type>::value,
    ::yampi::buffer<typename boost::range_value<ContiguousRange>::type>
  >::type range_to_buffer(ContiguousRange& range)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  typename std::enable_if<
    ::yampi::has_predefined_datatype<typename boost::range_value<ContiguousRange const>::type>::value,
    ::yampi::buffer<typename boost::range_value<ContiguousRange const>::type>
  >::type range_to_buffer(ContiguousRange const& range)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<typename boost::range_value<ContiguousRange>::type>
  range_to_buffer(
    ContiguousRange& range,
    ::yampi::predefined_datatype<typename boost::range_value<ContiguousRange>::type> const&)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<typename boost::range_value<ContiguousRange const>::type>
  range_to_buffer(
    ContiguousRange const& range,
    ::yampi::predefined_datatype<typename boost::range_value<ContiguousRange const>::type> const&)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range))))
  { return ::yampi::make_buffer(std::begin(range), std::end(range)); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<typename boost::range_value<ContiguousRange>::type>
  range_to_buffer(ContiguousRange& range, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range), datatype)))
  { return ::yampi::make_buffer(std::begin(range), std::end(range), datatype); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<typename boost::range_value<ContiguousRange const>::type>
  range_to_buffer(ContiguousRange const& range, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_buffer(std::begin(range), std::end(range), datatype)))
  { return ::yampi::make_buffer(std::begin(range), std::end(range), datatype); }
}


# undef YAMPI_is_nothrow_swappable

#endif
