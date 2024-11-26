#ifndef YAMPI_IMMEDIATE_REQUEST_HPP
# define YAMPI_IMMEDIATE_REQUEST_HPP

# include <type_traits>

# include <mpi.h>

# include <yampi/request_base.hpp>


namespace yampi
{
  class immediate_request
    : public ::yampi::request_base
  {
    using base_type = ::yampi::request_base;

   public:
    using base_type::base_type;

    immediate_request() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    immediate_request(immediate_request const&) = delete;
    immediate_request& operator=(immediate_request const&) = delete;
    immediate_request(immediate_request&&) = default;
    immediate_request& operator=(immediate_request&&) = default;
    ~immediate_request() noexcept = default;
  };
}


#endif

