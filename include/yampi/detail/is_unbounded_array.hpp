#ifndef YAMPI_DETAIL_IS_UNBOUNDED_ARRAY_HPP
# define YAMPI_DETAIL_IS_UNBOUNDED_ARRAY_HPP

# include <cstddef>
# include <type_traits>


namespace yampi
{
  namespace detail
  {
    template <typename T>
    struct is_unbounded_array
      : std::false_type
    { };

    template <typename T>
    struct is_unbounded_array<T[]>
      : std::true_type
    { };
  }
}


#endif

