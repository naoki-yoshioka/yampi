#ifndef YAMPI_DETAIL_IS_UNBOUNDED_ARRAY_HPP
# define YAMPI_DETAIL_IS_UNBOUNDED_ARRAY_HPP

# include <boost/config.hpp>

# include <cstddef>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_true_type std::true_type
#   define YAMPI_false_type std::false_type
# else
#   define YAMPI_true_type boost::true_type
#   define YAMPI_false_type boost::false_type
# endif


namespace yampi
{
  namespace detail
  {
    template <typename T>
    struct is_unbounded_array
      : YAMPI_false_type
    { };

    template <typename T>
    struct is_unbounded_array<T[]>
      : YAMPI_true_type
    { };
  }
}


# undef YAMPI_false_type
# undef YAMPI_true_type

#endif

