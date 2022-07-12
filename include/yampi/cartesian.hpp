#ifndef YAMPI_CARTESIAN_HPP
# define YAMPI_CARTESIAN_HPP

# include <vector>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/topology.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/rank.hpp>


namespace yampi
{
  class cartesian_shift
  {
    int direction_;
    int displacement_;

   public:
    cartesian_shift(int const direction, int const displacement)
      : direction_(direction), displacement_(displacement)
    { }

    int const& direction() const { return direction_; }
    int const& displacement() const { return displacement_; }
  };

  class cartesian
    : public ::yampi::topology
  {
    typedef ::yampi::topology base_type;

   public:
    cartesian() = delete;
    cartesian(cartesian const&) = delete;
    cartesian& operator=(cartesian const&) = delete;
    cartesian(cartesian&&) = default;
    cartesian& operator=(cartesian&&) = default;
    ~cartesian() = default;

    using base_type::base_type;

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    cartesian(
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      ::yampi::environment const& environment)
      : base_type(
          create(
            environment, old_communicator, size_first, size_last,
            is_periodic_first, true, environment))
    { }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    cartesian(
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      bool const is_reorderable,
      ::yampi::environment const& environment)
      : base_type(
          create(
            old_communicator, size_first, size_last,
            is_periodic_first, is_reorderable, environment))
    { }

    template <typename ContiguousIterator>
    cartesian(
      cartesian const& other,
      ContiguousIterator const remains_first, ContiguousIterator const remains_last,
      ::yampi::environment const& environment)
      : base_type(make_subcommunicator(other, remains_first, remains_last, environment))
    { }

   private:
    template <typename ContiguousIterator1, typename ContiguousIterator2>
    MPI_Comm create(
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      bool const is_reorderable,
      ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator2 must be the same to bool");

      std::vector<int> my_is_periodic(is_periodic_first, is_periodic_first + (size_last - size_first));

      MPI_Comm result;
      int const error_code
        = MPI_Cart_create(
            old_communicator.mpi_comm(),
            size_last-size_first, std::addressof(*size_first),
            std::addressof(my_is_periodic.front()),
            static_cast<int>(is_reorderable),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::cartesian::create", environment);
    }

    template <typename ContiguousIterator>
    MPI_Comm make_subcommunicator(
      cartesian const& other,
      ContiguousIterator const remains_first, ContiguousIterator const remains_last,
      ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator must be the same to bool");

      std::vector<int> my_remains(remains_first, remains_last);

      MPI_Comm result;
      int const error_code
        = MPI_Cart_sub(
            other.communicator().mpi_comm(), std::addressof(my_remains.front()),
            std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::cartesian::make_subcommunicator", environment);
    }

   public:
    using base_type::reset;

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    void reset(
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      ::yampi::environment const& environment)
    {
      communicator_.free(environment);
      communicator_.mpi_comm(
        create(
          old_communicator, size_first, size_last, is_periodic_first, true,
          environment));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    void reset(
      ::yampi::communicator const& old_communicator,
      ContiguousIterator1 const size_first, ContiguousIterator1 const size_last,
      ContiguousIterator2 const is_periodic_first,
      bool const is_reorderable,
      ::yampi::environment const& environment)
    {
      communicator_.free(environment);
      communicator_.mpi_comm(
        create(
          old_communicator, size_first, size_last, is_periodic_first, is_reorderable,
          environment));
    }

    template <typename ContiguousIterator>
    void reset(
      cartesian const& other,
      ContiguousIterator const remains_first, ContiguousIterator const remains_last,
      ::yampi::environment const& environment)
    {
      communicator_.free(environment);
      communicator_.mpi_comm(
        make_subcommunicator(other, remains_first, remains_last, environment));
    }

    int dimension(::yampi::environment const& environment) const
    {
      int result;
      int const error_code
        = MPI_Cartdim_get(communicator_.mpi_comm(), std::addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::cartesian::dimension", environment);
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename ContiguousIterator3>
    void topology_information(
      ContiguousIterator1 const size_out_first, ContiguousIterator1 const size_out_last,
      ContiguousIterator2 const is_periodic_out,
      ContiguousIterator3 const coordinates_out,
      ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator1>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator2>::value_type>::type,
           bool>::value),
        "Value type of ContiguousIterator2 must be the same to bool");
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator3>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator3 must be the same to int");

      std::vector<int> my_is_periodic(size_out_last - size_out_first);

      int const error_code
        = MPI_Cart_get(
            communicator_.mpi_comm(),
            size_out_last - size_out_first, std::addressof(*size_out_first),
            std::addressof(my_is_periodic.front()),
            std::addressof(*coordinates_out));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::topology_information", environment);
      std::transform(
        std::begin(my_is_periodic), std::end(my_is_periodic), is_periodic_out);
    }

    template <typename ContiguousIterator>
    ::yampi::rank rank(
      ContiguousIterator const coordinates_first, ::yampi::environment const& environment) const
    {
      static_assert(
        (std::is_same<
           typename std::remove_cv<
             typename std::iterator_traits<ContiguousIterator>::value_type>::type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");

      int mpi_rank;
      int const error_code
        = MPI_Cart_rank(
            communicator_.mpi_comm(), std::addressof(*coordinates_first),
            std::addressof(mpi_rank));
      return error_code == MPI_SUCCESS
        ? ::yampi::rank(mpi_rank)
        : throw ::yampi::error(error_code, "yampi::cartesian::rank", environment);
    }

    template <typename ContiguousIterator>
    void coordinates(
      ::yampi::rank const rank,
      ContiguousIterator const coordinates_out_first,
      ContiguousIterator const coordinates_out_last,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           int>::value),
        "Value type of ContiguousIterator1 must be the same to int");

      int const error_code
        = MPI_Cart_coords(
            communicator_.mpi_comm(), rank.mpi_rank(),
            coordinates_out_last - coordinates_out_first,
            std::addressof(*coordinates_out_first));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::cartesian::coordinates", environment);
    }
  };

  inline void swap(::yampi::cartesian& lhs, ::yampi::cartesian& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#endif

