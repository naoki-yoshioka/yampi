#ifndef YAMPI_STARTABLE_REQUEST_HPP
# define YAMPI_STARTABLE_REQUEST_HPP

# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/request_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class startable_request
    : public ::yampi::request_base
  {
    using base_type = ::yampi::request_base;

   public:
    using base_type::base_type;

    startable_request() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    startable_request(startable_request const&) = delete;
    startable_request& operator=(startable_request const&) = delete;
    startable_request(startable_request&&) = default;
    startable_request& operator=(startable_request&&) = default;

   protected:
    ~startable_request() noexcept = default;

   public:
    void start(::yampi::environment const& environment)
    {
      auto const error_code = MPI_Start(std::addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::startable_request::start", environment};
    }
  };
}


#endif

