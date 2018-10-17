#ifndef YAMPI_COMPARE_HPP
# define YAMPI_COMPARE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/communicator.hpp>
# include <yampi/group.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class communicators_are
    : int
  {
    identical = MPI_IDENT, congruent = MPI_CONGRUENT,
    similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
  };

  enum class groups_are
    : int
  {
    identical = MPI_IDENT,
    similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
  };

#   define YAMPI_COMMUNICATORS_ARE ::yampi::communicators_are
#   define YAMPI_GROUPS_ARE ::yampi::groups_are
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  namespace communicators_are
  {
    enum communicators_are_
    {
      identical = MPI_IDENT, congruent = MPI_CONGRUENT,
      similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
    };
  }

  namespace groups_are
  {
    enum groups_are_
    {
      identical = MPI_IDENT,
      similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
    };
  }

#   define YAMPI_COMMUNICATORS_ARE ::yampi::communicators_are::communicators_are_
#   define YAMPI_GROUPS_ARE ::yampi::groups_are::groups_are_
# endif // BOOST_NO_CXX11_SCOPED_ENUMS


  inline YAMPI_COMMUNICATORS_ARE compare(
    ::yampi::communicator const& lhs, ::yampi::communicator const& rhs,
    ::yampi::environment const& environment)
  {
    int result;
    int const error_code = MPI_Comm_compare(lhs.mpi_comm(), rhs.mpi_comm(), &result);
    return error_code == MPI_SUCCESS
      ? static_cast<YAMPI_COMMUNICATORS_ARE>(result)
      : throw ::yampi::error(error_code, "yampi::compare", environment);
  }

  inline YAMPI_GROUPS_ARE compare(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  {
    int result;
    int const error_code = MPI_Group_compare(lhs.mpi_group(), rhs.mpi_group(), &result);
    return error_code == MPI_SUCCESS
      ? static_cast<YAMPI_GROUPS_ARE>(result)
      : throw ::yampi::error(error_code, "yampi::compare", environment);
  }

# undef YAMPI_COMMUNICATORS_ARE
# undef YAMPI_GROUPS_ARE


  inline bool is_identical(
    ::yampi::communicator const& lhs, ::yampi::communicator const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::identical; }

  inline bool is_identical(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::identical; }

  inline bool is_congruent(
    ::yampi::communicator const& lhs, ::yampi::communicator const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::congruent; }

  inline bool is_similar(
    ::yampi::communicator const& lhs, ::yampi::communicator const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::similar; }

  inline bool is_similar(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::similar; }

  inline bool is_unequal(
    ::yampi::communicator const& lhs, ::yampi::communicator const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::unequal; }

  inline bool is_unequal(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::unequal; }
}


#endif

