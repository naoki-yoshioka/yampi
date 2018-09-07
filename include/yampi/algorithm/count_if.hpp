#ifndef YAMPI_ALGORITHM_COUNT_IF_HPP
# define YAMPI_ALGORITHM_COUNT_IF_HPP

# include <boost/config.hpp>

# include <iterator>
# include <algorithm>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <boost/optional.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/reduce.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/binary_operation.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename Value, typename UnaryPredicate>
    inline
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count_if(
      ::yampi:communicator const& communicator, ::yampi::rank const root,
      ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      UnaryPredicate unary_predicate)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result
        = std::count_if(buffer.data(), buffer.data() + buffer.count_if(), unary_predicate);

      ::yampi::reduce const reducer(communicator, root);

      if (communicator.rank(environment) == root)
      {
        reducer.call(
          environment, ::yampi::make_buffer(result), YAMPI_addressof(result),
          ::yampi::binary_operation(::yampi::plus_t()));
        return boost::make_optional(result);
      }

      reducer.call(
        environment, ::yampi::make_buffer(result),
        ::yampi::binary_operation(::yampi::plus_t()));
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline typename std::iterator_traits<Value const*>::difference_type
    count_if(
      ::yampi:communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      UnaryPredicate unary_predicate)
    {
      return ::yampi::all_reduce(
        communicator, environment,
        ::yampi::make_buffer(
          std::count_if(buffer.data(), buffer.data() + buffer.count_if(), unary_predicate)),
        ::yampi::binary_operation(::yampi::plus_t()));
    }
# if MPI_VERSION >= 3


    template <typename Value, typename UnaryPredicate>
    inline
    boost::optional<typename std::iterator_traits<Value const*>::difference_type>
    count_if(
      ::yampi:communicator const& communicator, ::yampi::rank const root,
      ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::request& request,
      UnaryPredicate unary_predicate)
    {
      typedef typename std::iterator_traits<Value const*>::difference_type count_type;
      count_type result
        = std::count_if(buffer.data(), buffer.data() + buffer.count_if(), unary_predicate);

      ::yampi::reduce const reducer(communicator, root);

      if (communicator.rank(environment) == root)
      {
        reducer.call(
          environment, ::yampi::make_buffer(result), YAMPI_addressof(result),
          request,
          ::yampi::binary_operation(::yampi::plus_t()));
        return boost::make_optional(result);
      }

      reducer.call(
        environment, ::yampi::make_buffer(result),
        request,
        ::yampi::binary_operation(::yampi::plus_t()));
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline typename std::iterator_traits<Value const*>::difference_type
    count_if(
      ::yampi:communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::request& request,
      UnaryPredicate unary_predicate)
    {
      return ::yampi::all_reduce(
        communicator, environment,
        ::yampi::make_buffer(
          std::count_if(buffer.data(), buffer.data() + buffer.count_if(), unary_predicate)),
        request,
        ::yampi::binary_operation(::yampi::plus_t()));
    }
# endif
  }
}


# undef YAMPI_addressof

#endif

