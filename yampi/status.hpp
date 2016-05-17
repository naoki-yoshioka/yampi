#ifndef YAMPI_STATUS_HPP
# define YAMPI_STATUS_HPP

# include <boost/config.hpp>

# include <cstddef>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <mpi.h>

# include <yampi/has_corresponding_mpi_data_type.hpp>
# include <yampi/mpi_data_type_of.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if
# endif


namespace yampi
{
  class count_value_undefined_error
    : public std::runtime_error
  {
   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    count_value_undefined_error()
      : std::runtime_error{"count in MPI_GET_COUNT is MPI_UNDEFINED"}
    { }
# else
    count_value_undefined_error()
      : std::runtime_error("count in MPI_GET_COUNT is MPI_UNDEFINED")
    { }
# endif
  };

  template <typename Value>
  class status
  {
    MPI_Status stat_;

   public:
    BOOST_DELETED_FUNCTION(status())

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    explicit status(MPI_Status const& stat) BOOST_NOEXCEPT_OR_NOTHROW : stat_{stat} { }
# else
    explicit status(MPI_Status const& stat) BOOST_NOEXCEPT_OR_NOTHROW : stat_(stat)} { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    status(status const&) = default;
    status& operator=(status const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    status(status&&) = default;
    status& operator=(status&&) = default;
#   endif
    ~status() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::rank source() const BOOST_NOEXCEPT_OR_NOTHROW { return ::yampi::rank{stat.MPI_SOURCE}; }
    ::yampi::tag tag() const BOOST_NOEXCEPT_OR_NOTHROW { return ::yampi::tag{stat.MPI_TAG}; }
    void throw_error() const { throw ::yampi::error{stat_.MPI_ERROR, "yampi::status::throw_error"}; }
# else
    ::yampi::rank source() const BOOST_NOEXCEPT_OR_NOTHROW { return ::yampi::rank(stat.MPI_SOURCE); }
    ::yampi::tag tag() const BOOST_NOEXCEPT_OR_NOTHROW { return ::yampi::tag(stat.MPI_TAG); }
    void throw_error() const { throw ::yampi::error(stat_.MPI_ERROR, "yampi::status::throw_error"); }
# endif

    typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, std::size_t>::type
    message_length() const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto length = int{};
#   else
      auto length = int();
#   endif

      auto const error_code = MPI_GET_COUNT(&stat_, ::yampi::mpi_data_type_of<Value>::value, &length);
# else
      int length;

      int const error_code = MPI_GET_COUNT(&stat_, ::yampi::mpi_data_type_of<Value>::value, &length);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::status::message_length"};

      if (count == MPI_UNDEFINED)
        throw ::yampi::count_value_undefiend_error{};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::status::message_length");

      if (count == MPI_UNDEFINED)
        throw ::yampi::count_value_undefiend_error();
# endif

      return length;
    }

    bool empty() const
    { return stat_.MPI_TAG == MPI_ANY_TAG && stat_.MPI_SOURCE == MPI_ANY_SOURCE && stat_.MPI_ERROR == MPI_SUCCESS; }
  };


  class ignore_status_t { };

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  auto BOOST_CONSTEXPR_OR_CONST ignore_status = ignore_status_t{};
#   else
  auto BOOST_CONSTEXPR_OR_CONST ignore_status = ignore_status_t();
#   endif
# else
  ignore_status_t BOOST_CONSTEXPR_OR_CONST ignore_status;
# endif
}


# undef YAMPI_enable_if

#endif

