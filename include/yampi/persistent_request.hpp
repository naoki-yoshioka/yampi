#ifndef YAMPI_PERSISTENT_REQUEST_HPP
# define YAMPI_PERSISTENT_REQUEST_HPP

# include <type_traits>

# include <mpi.h>

# include <yampi/startable_request.hpp>


namespace yampi
{
  class persistent_request
    : public ::yampi::startable_request
  {
    using base_type = ::yampi::startable_request;

   public:
    using base_type::base_type;

    persistent_request() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    persistent_request(persistent_request const&) = delete;
    persistent_request& operator=(persistent_request const&) = delete;
    persistent_request(persistent_request&&) = default;
    persistent_request& operator=(persistent_request&&) = default;
    ~persistent_request() noexcept = default;
  };
}


#endif

