#ifndef YAMPI_DATATYPE_BASE_HPP
# define YAMPI_DATATYPE_BASE_HPP

# include <mpi.h>


namespace yampi
{
  template <typename Derived>
  class datatype_base
  {
   public:
    bool is_null() const noexcept { return derived().do_is_null(); }

    MPI_Datatype mpi_datatype() const noexcept { return derived().do_mpi_datatype(); }

   protected:
    ~datatype_base() noexcept = default;

    Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    Derived const& derived() const noexcept { return static_cast<Derived const&>(*this); }
  }; // class datatype_base<Derived>
} // namespace yampi

#endif // YAMPI_DATATYPE_BASE_HPP
