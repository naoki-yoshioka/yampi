#ifndef YAMPI_PARTITIONED_BUFFER_HPP
# define YAMPI_PARTITIONED_BUFFER_HPP

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

# if MPI_VERSION >= 4
namespace yampi
{
  template <typename T, typename Enable = void>
  class partitioned_buffer
  {
    T* data_;
    int num_partitions_;
    ::yampi::count count_;
    ::yampi::datatype* datatype_ptr_;

   public:
    template <typename ContiguousIterator>
    partitioned_buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      int const num_partitions, ::yampi::datatype const& datatype)
      noexcept(noexcept(*first) and noexcept(last - first))
      : data_{std::addressof(*first)},
        num_partitions_{num_partitions},
        count_{static_cast<MPI_Count>(last - first) / static_cast<MPI_Count>(num_partitions)},
        datatype_ptr_{const_cast< ::yampi::datatype* >(std::addressof(datatype))}
    {
      static_assert(
        std::is_same<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>, T>::value,
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
      assert(static_cast<MPI_Count>(num_partitions_) * count_.mpi_count() == static_cast<MPI_Count>(last - first));
    }

    bool operator==(partitioned_buffer const& other) const noexcept
    { return data_ == other.data_ and num_partitions_ == other.num_partitions_ and count_ == other.count_ and *datatype_ptr_ == *other.datatype_ptr_; }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }
    int const& num_partitions() const noexcept { return num_partitions_; }
    ::yampi::count const& count() const noexcept { return count_; }
    ::yampi::datatype const& datatype() const noexcept { return *datatype_ptr_; }

    void swap(partitioned_buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(num_partitions_, other.num_partitions_);
      swap(count_, other.count_);
      swap(datatype_ptr_, other.datatype_ptr_);
    }
  }; // class partitioned_buffer<T, Enable>

  template <typename T>
  class partitioned_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>
  {
    T* data_;
    int num_partitions_;
    ::yampi::count count_;

   public:
    template <typename ContiguousIterator>
    partitioned_buffer(ContiguousIterator const first, ContiguousIterator const last, int const num_partitions)
      noexcept(noexcept(*first) and noexcept(last - first))
      : data_{std::addressof(*first)},
        num_partitions_{num_partitions},
        count_{static_cast<MPI_Count>(last - first) / static_cast<MPI_Count>(num_partitions)}
    {
      static_assert(
        std::is_same<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>, T>::value,
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
      assert(static_cast<MPI_Count>(num_partitions_) * count_.mpi_count() == static_cast<MPI_Count>(last - first));
    }

    bool operator==(partitioned_buffer const& other) const noexcept
    { return data_ == other.data_ and num_partitions_ == other.num_partitions_ and count_ == other.count_; }

    T* data() noexcept { return data_; }
    T const* data() const noexcept { return data_; }
    int const& num_partitions() const noexcept { return num_partitions_; }
    ::yampi::count const& count() const noexcept { return count_; }
    ::yampi::predefined_datatype<T> datatype() const noexcept { return ::yampi::predefined_datatype<T>(); }

    void swap(partitioned_buffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(num_partitions_, other.num_partitions_);
      swap(count_, other.count_);
    }
  }; // class partitioned_buffer<T, typename std::enable_if< ::yampi::has_predefined_datatype<T>::value >::type>

  template <typename T>
  inline bool operator!=(::yampi::partitioned_buffer<T> const& lhs, ::yampi::partitioned_buffer<T> const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::partitioned_buffer<T>& lhs, ::yampi::partitioned_buffer<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  template <typename ContiguousIterator>
  inline
  typename std::enable_if_t<
    ::yampi::has_predefined_datatype<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>::value,
    ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>>
  make_partitioned_buffer(ContiguousIterator const first, ContiguousIterator const last, int const num_partitions)
  noexcept(
    noexcept(
      ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>(
        first, last, num_partitions)))
  {
    using result_type
      = ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>;
    return result_type{first, last, num_partitions};
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>
  make_partitioned_buffer(
    ContiguousIterator const first, ContiguousIterator const last, int const num_partitions,
    ::yampi::predefined_datatype<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>> const&)
  noexcept(
    noexcept(
      ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>(
        first, last, num_partitions)))
  {
    using result_type
      = ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>;
    return result_type{first, last, num_partitions};
  }

  template <typename ContiguousIterator>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>
  make_partitioned_buffer(
    ContiguousIterator const first, ContiguousIterator const last,
    int const num_partitions, ::yampi::datatype const& datatype)
  noexcept(
    noexcept(
      ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>(
        first, last, num_partitions, datatype)))
  {
    using result_type
      = ::yampi::partitioned_buffer<std::remove_cv_t<typename std::iterator_traits<ContiguousIterator>::value_type>>;
    return result_type{first, last, num_partitions, datatype};
  }

  template <typename ContiguousRange>
  inline
  std::enable_if_t<
    ::yampi::has_predefined_datatype<std::remove_cv_t<typename boost::range_value<ContiguousRange>::type>>::value,
    ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange>::type>>>
  range_to_partitioned_buffer(ContiguousRange& range, int const num_partitions)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions)))
  { using std::begin; using std::end; return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions); }

  template <typename ContiguousRange>
  inline
  std::enable_if_t<
    ::yampi::has_predefined_datatype<
      std::remove_cv_t<typename boost::range_value<ContiguousRange const>::type>>::value,
    ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange const>::type>>>
  range_to_partitioned_buffer(ContiguousRange const& range, int const num_partitions)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions)))
  { using std::begin; using std::end; return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions); }

  template <typename ContiguousRange>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange>::type>>
  range_to_partitioned_buffer(
    ContiguousRange& range, int const num_partitions,
    ::yampi::predefined_datatype<std::remove_cv_t<typename boost::range_value<ContiguousRange>::type>> const&)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions)))
  { using std::begin; using std::end ;return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions); }

  template <typename ContiguousRange>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange const>::type>>
  range_to_partitioned_buffer(
    ContiguousRange const& range, int const num_partitions,
    ::yampi::predefined_datatype<std::remove_cv<typename boost::range_value<ContiguousRange const>::type>> const&)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions)))
  { using std::begin; using std::end; return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions); }

  template <typename ContiguousRange>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange>::type>>
  range_to_partitioned_buffer(ContiguousRange& range, int const num_partitions, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions, datatype)))
  { using std::begin; using std::end; return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions, datatype); }

  template <typename ContiguousRange>
  inline
  ::yampi::partitioned_buffer<std::remove_cv_t<typename boost::range_value<ContiguousRange const>::type>>
  range_to_partitioned_buffer(ContiguousRange const& range, int const num_partitions, ::yampi::datatype const& datatype)
    noexcept(noexcept(::yampi::make_partitioned_buffer(std::begin(range), std::end(range), num_partitions, datatype)))
  { using std::begin; using std::end; return ::yampi::make_partitioned_buffer(begin(range), end(range), num_partitions, datatype); }
}
# endif // MPI_VERSION >= 4

# undef YAMPI_is_nothrow_swappable

#endif
