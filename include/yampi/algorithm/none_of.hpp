#ifndef YAMPI_ALGORITHM_NONE_OF_HPP
# define YAMPI_ALGORITHM_NONE_OF_HPP

# include <boost/config.hpp>

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
    inline boost::optional<bool> none_of(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const root,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      bool result
        = std::none_of(buffer.data(), buffer.data() + buffer.count(), unary_predicate);

      ::yampi::reduce const reducer(root, communicator);

      ::yampi::datatype bool_datatype(::yampi::bool_datatype_t());
      if (communicator.rank(environment) == root)
      {
        reducer.call(
          ::yampi::make_buffer(result, bool_datatype), YAMPI_addressof(result),
          ::yampi::binary_operation(::yampi::logical_and_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        ::yampi::make_buffer(result, bool_datatype),
        ::yampi::binary_operation(::yampi::logical_and_t()), environment);
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline bool none_of(
      ::yampi::buffer<Value> const& buffer,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        ::yampi::make_buffer(
          std::none_of(buffer.data(), buffer.data() + buffer.count(), unary_predicate),
          ::yampi::datatype(::yampi::bool_datatype_t())),
        ::yampi::binary_operation(::yampi::logical_and_t()),
        communicator, environment);
    }
# if MPI_VERSION >= 3


    template <typename Value, typename UnaryPredicate>
    inline boost::optional<bool> none_of(
      ::yampi::request& request,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::rank const root,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      bool result
        = std::none_of(buffer.data(), buffer.data() + buffer.count(), unary_predicate);

      ::yampi::reduce const reducer(root, communicator);

      ::yampi::datatype bool_datatype(::yampi::bool_datatype_t());
      if (communicator.rank(environment) == root)
      {
        reducer.call(
          request,
          ::yampi::make_buffer(result, bool_datatype), YAMPI_addressof(result),
          ::yampi::binary_operation(::yampi::logical_and_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        request,
        ::yampi::make_buffer(result, bool_datatype),
        ::yampi::binary_operation(::yampi::logical_and_t()), environment);
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline bool none_of(
      ::yampi::request& request,
      ::yampi::buffer<Value> const& buffer,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::all_reduce(
        request,
        ::yampi::make_buffer(
          std::none_of(buffer.data(), buffer.data() + buffer.count(), unary_predicate),
          ::yampi::datatype(::yampi::bool_datatype_t())),
        ::yampi::binary_operation(::yampi::logical_and_t()),
        communicator, environment);
    }
# endif
  }
}


# undef YAMPI_addressof

#endif

