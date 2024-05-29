#ifndef YAMPI_ALGORITHM_ANY_OF_HPP
# define YAMPI_ALGORITHM_ANY_OF_HPP

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
    template <typename Value, typename UnaryPredicate>
    inline boost::optional<bool> any_of(
      ::yampi::buffer<Value> const buffer,
      UnaryPredicate unary_predicate, ::yampi::rank const root,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const buffer_size = buffer.count().mpi_count();
# else // MPI_VERSION >= 4
      auto const buffer_size = buffer.count();
# endif // MPI_VERSION >= 4
      bool result = std::any_of(buffer.data(), buffer.data() + buffer_size, unary_predicate);

      if (communicator.rank(environment) == root)
      {
        ::yampi::reduce(
          ::yampi::make_buffer(result), std::addressof(result),
          ::yampi::binary_operation(::yampi::logical_or_t()), root, communicator, environment);
        return boost::make_optional(result);
      }

      ::yampi::reduce(
        ::yampi::make_buffer(result),
        ::yampi::binary_operation(::yampi::logical_or_t()), root, communicator, environment);
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline bool any_of(
      ::yampi::buffer<Value> const buffer,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const buffer_size = buffer.count().mpi_count();
# else // MPI_VERSION >= 4
      auto const buffer_size = buffer.count();
# endif // MPI_VERSION >= 4
      return ::yampi::all_reduce(
        ::yampi::make_buffer(std::any_of(buffer.data(), buffer.data() + buffer_size, unary_predicate)),
        ::yampi::binary_operation(::yampi::logical_or_t()),
        communicator, environment);
    }
    /*
# if MPI_VERSION >= 3

    template <typename Value, typename UnaryPredicate>
    inline boost::optional<bool> any_of(
      ::yampi::request& request,
      ::yampi::buffer<Value> const buffer,
      UnaryPredicate unary_predicate, ::yampi::rank const& root,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const buffer_size = buffer.count().mpi_count();
# else // MPI_VERSION >= 4
      auto const buffer_size = buffer.count();
# endif // MPI_VERSION >= 4
      bool result
        = std::any_of(buffer.data(), buffer.data() + buffer_size, unary_predicate);

      ::yampi::reduce const reducer(root, communicator);

      if (communicator.rank(environment) == root)
      {
        reducer.call(
          request,
          ::yampi::make_buffer(result), std::addressof(result),
          ::yampi::binary_operation(::yampi::logical_or_t()), environment);
        return boost::make_optional(result);
      }

      reducer.call(
        request,
        ::yampi::make_buffer(result),
        ::yampi::binary_operation(::yampi::logical_or_t()), environment);
      return boost::none;
    }

    template <typename Value, typename UnaryPredicate>
    inline bool any_of(
      ::yampi::request& request,
      ::yampi::buffer<Value> const buffer,
      UnaryPredicate unary_predicate,
      ::yampi:communicator const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const buffer_size = buffer.count().mpi_count();
# else // MPI_VERSION >= 4
      auto const buffer_size = buffer.count();
# endif // MPI_VERSION >= 4
      return ::yampi::all_reduce(
        request,
        ::yampi::make_buffer(std::any_of(buffer.data(), buffer.data() + buffer_size, unary_predicate)),
        ::yampi::binary_operation(::yampi::logical_or_t()),
        communicator, environment);
    }
# endif
*/
  }
}


#endif

