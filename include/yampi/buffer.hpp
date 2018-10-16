#ifndef YAMPI_BUFFER_HPP
# define YAMPI_BUFFER_HPP

# include <boost/config.hpp>

# include <cassert>
# ifdef BOOST_NO_CXX11_NULLPTR
#   include <cstddef>
# endif
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
#   include <boost/type_traits/is_same.hpp>
#   include <boost/utility/enable_if.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif
# include <boost/range/value_type.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/datatype.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
#   define YAMPI_is_same std::is_same
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_remove_cv boost::remove_cv
#   define YAMPI_is_same boost::is_same
#   define YAMPI_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif


namespace yampi
{
  template <typename T>
  class buffer
  {
    T* data_;
    int count_;
    ::yampi::datatype& datatype_;

   public:
    buffer(T& value, ::yampi::datatype& datatype)
      : data_(YAMPI_addressof(value)), count_(1),
        datatype_(datatype)
    { }

    buffer(T& value, ::yampi::datatype const& datatype)
      : data_(YAMPI_addressof(value)), count_(1),
        datatype_(const_cast< ::yampi::datatype& >(datatype))
    { }

    buffer(T const& value, ::yampi::datatype& datatype)
      : data_(const_cast<T*>(YAMPI_addressof(value))), count_(1),
        datatype_(datatype)
    { }

    buffer(T const& value, ::yampi::datatype const& datatype)
      : data_(const_cast<T*>(YAMPI_addressof(value))), count_(1),
        datatype_(const_cast< ::yampi::datatype& >(datatype))
    { }

    template <typename ContiguousIterator>
    buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype& datatype)
      : data_(const_cast<T*>(YAMPI_addressof(*first))),
        count_(last-first),
        datatype_(datatype)
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           T>::value),
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    template <typename ContiguousIterator>
    buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype const& datatype)
      : data_(const_cast<T*>(YAMPI_addressof(*first))),
        count_(last-first),
        datatype_(const_cast< ::yampi::datatype& >(datatype))
    {
      static_assert(
        (YAMPI_is_same<
           typename YAMPI_remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           T>::value),
        "T must be tha same to value_type of ContiguousIterator");
      assert(last >= first);
    }

    T* data() { return data_; }
    T const* data() const { return data_; }
    int const& count() const { return count_; }
    ::yampi::datatype const& datatype() const { return datatype_; }

    void swap(buffer& other)
      BOOST_NOEXCEPT_IF(
        ::yampi::utility::is_nothrow_swappable<T*>::value
        and ::yampi::utility::is_nothrow_swappable<int>::value
        and ::yampi::utility::is_nothrow_swappable< ::yampi::datatype& >::value)
    {
      using std::swap;
      swap(data_, other.data_);
      swap(count_, other.count_);
      swap(datatype_, other.datatype_);
    }
  };


  template <typename T>
  inline bool operator==(::yampi::buffer<T> const& lhs, ::yampi::buffer<T> const& rhs)
  {
    return lhs.data() == rhs.data() and lhs.count() == rhs.count()
      and lhs.datatype() == rhs.datatype();
  }

  template <typename T>
  inline bool operator!=(::yampi::buffer<T> const& lhs, ::yampi::buffer<T> const& rhs)
  { return not (lhs == rhs); }

  template <typename T>
  inline void swap(::yampi::buffer<T>& lhs, ::yampi::buffer<T>& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T& value, ::yampi::datatype const& datatype)
  { return ::yampi::buffer<T>(value, datatype); }

  template <typename T>
  inline ::yampi::buffer<T> make_buffer(T const& value, ::yampi::datatype const& datatype)
  { return ::yampi::buffer<T>(value, datatype); }

  template <typename ContiguousIterator>
  inline
  ::yampi::buffer<
    typename YAMPI_remove_cv<
      typename std::iterator_traits<ContiguousIterator>::value_type>::type>
  make_buffer(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::datatype const& datatype)
  {
    typedef
      ::yampi::buffer<
        typename YAMPI_remove_cv<
          typename std::iterator_traits<ContiguousIterator>::value_type>::type>
      result_type;
    return result_type(first, last, datatype);
  }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename YAMPI_remove_cv<
      typename boost::range_value<ContiguousRange>::type>::type>
  range_to_buffer(ContiguousRange& range, ::yampi::datatype const& datatype)
  { return ::yampi::make_buffer(boost::begin(range), boost::end(range), datatype); }

  template <typename ContiguousRange>
  inline
  ::yampi::buffer<
    typename YAMPI_remove_cv<
      typename boost::range_value<ContiguousRange const>::type>::type>
  range_to_buffer(ContiguousRange const& range, ::yampi::datatype const& datatype)
  { return ::yampi::make_buffer(boost::begin(range), boost::end(range), datatype); }
}


# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_enable_if
# undef YAMPI_is_same
# undef YAMPI_remove_cv

#endif
