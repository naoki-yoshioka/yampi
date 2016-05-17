#ifndef YAMPI_IS_CONTIGUOUS_ITERATOR_HPP
# define YAMPI_IS_CONTIGUOUS_ITERATOR_HPP

# include <boost/config.hpp>

# include <cstddef>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif
# include <vector>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_true_type std::true_type
#   define YAMPI_false_type std::false_type
# else
#   define YAMPI_true_type boost::true_type
#   define YAMPI_false_type boost::false_type
# endif


namespace yampi
{
  template <typename Range>
  struct is_contiguous_iterator
    : YAMPI_false_type
  { };

  template <typename Value, typename Allocator>
  struct is_contiguous_iterator<typename std::vector<Value, Allocator>::iterator>
    : YAMPI_true_type
  { };

  template <typename Value, typename Allocator>
  struct is_contiguous_iterator<typename std::vector<Value, Allocator>::const_iterator>
    : YAMPI_true_type
  { };

  template <typename Value>
  struct is_contiguous_iterator<Value*>
    : YAMPI_true_type
  { };

  template <typename Value>
  struct is_contiguous_iterator<Value const*>
    : YAMPI_true_type
  { };
}


# undef YAMPI_true_type
# undef YAMPI_false_type

#endif

