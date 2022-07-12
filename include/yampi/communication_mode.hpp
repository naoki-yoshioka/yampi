#ifndef YAMPI_COMMUNICATION_MODE_HPP
# define YAMPI_COMMUNICATION_MODE_HPP


namespace yampi
{
  namespace mode
  {
    struct standard_communication_t { };
    struct buffered_communication_t { };
    struct synchronous_communication_t { };
    struct ready_communication_t { };

# if __cplusplus >= 201703L
    inline constexpr ::yampi::mode::standard_communication_t standard_communication{};
    inline constexpr ::yampi::mode::buffered_communication_t buffered_communication{};
    inline constexpr ::yampi::mode::synchronous_communication_t synchronous_communication{};
    inline constexpr ::yampi::mode::ready_communication_t ready_communication{};
# else
    constexpr ::yampi::mode::standard_communication_t standard_communication{};
    constexpr ::yampi::mode::buffered_communication_t buffered_communication{};
    constexpr ::yampi::mode::synchronous_communication_t synchronous_communication{};
    constexpr ::yampi::mode::ready_communication_t ready_communication{};
# endif
  }
}


#endif

