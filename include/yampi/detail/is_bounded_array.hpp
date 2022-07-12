#ifndef YAMPI_DETAIL_IS_BOUNDED_ARRAY_HPP
# define YAMPI_DETAIL_IS_BOUNDED_ARRAY_HPP

# include <cstddef>
# include <type_traits>


namespace yampi
{
  namespace detail
  {
    template <typename T>
    struct is_bounded_array
      : std::false_type
    { };

    template <typename T, std::size_t bound>
    struct is_bounded_array<T[bound]>
      : std::true_type
    { };
  }
}


#endif

