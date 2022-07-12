#ifndef YAMPI_ALGORITHM_COUNT_HPP
# define YAMPI_ALGORITHM_COUNT_HPP

# include <iterator>
# include <algorithm>
# include <memory>

# include <boost/optional.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/reduce.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/binary_operation.hpp>
//# if MPI_VERSION >= 3
//#   include <yampi/request.hpp>
//# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename Value>
    inline
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count(
      ::yampi::buffer<Value> const buffer,
      Value const& value, ::yampi::rank const root,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result = std::count(buffer.data(), buffer.data() + buffer.count(), value);

      if (communicator.rank(environment) == root)
      {
        ::yampi::reduce(
          ::yampi::make_buffer(result), std::addressof(result),
          ::yampi::binary_operation(::yampi::plus_t()), root, communicator, environment);
        return boost::make_optional(result);
      }

      ::yampi::reduce(
        ::yampi::make_buffer(result),
        ::yampi::binary_operation(::yampi::plus_t()), root, communicator, environment);
      return boost::none;
    }

    template <typename Value>
    inline typename std::iterator_traits<Value const*>::difference_type
    count(
      ::yampi::buffer<Value> const buffer,
      Value const& value,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        ::yampi::make_buffer(
          std::count(buffer.data(), buffer.data() + buffer.count(), value)),
        ::yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);
    }
    /*
# if MPI_VERSION >= 3

    template <typename Value>
    inline
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count(
      ::yampi::request& request,
      ::yampi::buffer<Value> const buffer,
      Value const& value, ::yampi::rank const& root,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result
        = std::count(buffer.data(), buffer.data() + buffer.count(), value);

      ::yampi::reduce const reducer(root, communicator);

      if (communicator.rank(environment) == root)
      {
        reducer.call(
          request,
          ::yampi::make_buffer(result), std::addressof(result),
          ::yampi::binary_operation(::yampi::plus_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        request,
        ::yampi::make_buffer(result),
        ::yampi::binary_operation(::yampi::plus_t()), environment);
      return boost::none;
    }

    template <typename Value>
    inline typename std::iterator_traits<Value const*>::difference_type
    count(
      ::yampi::request& request,
      ::yampi::buffer<Value> const buffer,
      Value const& value,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        request,
        ::yampi::make_buffer(std::count(buffer.data(), buffer.data() + buffer.count(), value)),
        ::yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);
    }
# endif
*/
  }
}


#endif

