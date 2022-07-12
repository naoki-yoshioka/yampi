#ifndef YAMPI_COMPARE_HPP
# define YAMPI_COMPARE_HPP

# include <mpi.h>

# include <yampi/communicator_base.hpp>
# include <yampi/group.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
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


  inline ::yampi::communicators_are compare(
    ::yampi::communicator_base const& lhs, ::yampi::communicator_base const& rhs,
    ::yampi::environment const& environment)
  {
    int result;
    int const error_code = MPI_Comm_compare(lhs.mpi_comm(), rhs.mpi_comm(), &result);
    return error_code == MPI_SUCCESS
      ? static_cast< ::yampi::communicators_are >(result)
      : throw ::yampi::error(error_code, "yampi::compare", environment);
  }

  inline ::yampi::groups_are compare(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  {
    int result;
    int const error_code = MPI_Group_compare(lhs.mpi_group(), rhs.mpi_group(), &result);
    return error_code == MPI_SUCCESS
      ? static_cast< ::yampi::groups_are >(result)
      : throw ::yampi::error(error_code, "yampi::compare", environment);
  }


  inline bool is_identical(
    ::yampi::communicator_base const& lhs, ::yampi::communicator_base const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::identical; }

  inline bool is_identical(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::identical; }

  inline bool is_congruent(
    ::yampi::communicator_base const& lhs, ::yampi::communicator_base const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::congruent; }

  inline bool is_similar(
    ::yampi::communicator_base const& lhs, ::yampi::communicator_base const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::similar; }

  inline bool is_similar(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::similar; }

  inline bool is_unequal(
    ::yampi::communicator_base const& lhs, ::yampi::communicator_base const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::unequal; }

  inline bool is_unequal(
    ::yampi::group const& lhs, ::yampi::group const& rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::groups_are::unequal; }
}


#endif

