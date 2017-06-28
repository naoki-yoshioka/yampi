#ifndef YAMPI_DERIVE_DATATYPE_HPP
# define YAMPI_DERIVE_DATATYPE_HPP

# include <boost/config.hpp>

# include <cstddef>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#   include <boost/preprocessor/arithmetic/dec.hpp>
#   include <boost/preprocessor/repetition/repeat.hpp>
#   include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#   include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#   ifndef YAMPI_NUM_DERIVE_DATATYPE_MEMBERS
#     define YAMPI_NUM_DERIVE_DATATYPE_MEMBERS 16
#   endif
# endif

# include <yampi/access.hpp>
# include <yampi/datatype.hpp>
# include <yampi/datatype_of.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define YAMPI_array std::array
# else
#   define YAMPI_array boost::array
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace derive_datatype_detail
  {
    template <std::size_t num_arguments, typename Value>
    void derive_datatype(
      YAMPI_array<int, num_arguments> const& blocklengths,
      YAMPI_array<MPI_Aint, num_arguments> const& displacements,
      YAMPI_array<MPI_Datatype, num_arguments> const& mpi_datatypes,
      Value const& value, MPI_Aint const& value_address)
    {
      MPI_Datatype temp;
      int const struct_error_code
        = MPI_Type_create_struct(
            num_arguments, blocklengths.data(), displacements.data(), mpi_datatypes.data(),
            YAMPI_addressof(temp));
      if (struct_error_code != MPI_SUCCESS)
        throw ::yampi::error(struct_error_code, "yampi::derive_datatype");

      MPI_Aint value_extent;
      int const address_error_code
        = MPI_Get_address(YAMPI_addressof(value)+1, YAMPI_addressof(value_extent));
      if (address_error_code != MPI_SUCCESS)
        throw ::yampi::error(address_error_code, "yampi::derive_datatype");

      value_extent -= value_address;

      MPI_Datatype result;
      int const resized_error_code
        = MPI_Type_create_resized(temp, 0, value_extent, YAMPI_addressof(result));
      if (resized_error_code != MPI_SUCCESS)
        throw ::yampi::error(resized_error_code, "yampi::derive_datatype");

      ::yampi::datatype_of<Value>::set(::yampi::datatype(result));
    }
  }

# ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
  namespace derive_datatype_detail
  {
    template <std::size_t num_arguments, typename Value, typename Member, typename... Members>
    inline void derive_datatype(
      YAMPI_array<int, num_arguments>& blocklengths,
      YAMPI_array<MPI_Aint, num_arguments>& displacements,
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,
      Value const& value, MPI_Aint const& value_address,
      Member const& member, Members const&... members)
    {
      MPI_Aint member_address;
      int const error_code
        = MPI_Get_address(YAMPI_addressof(member), YAMPI_addressof(member_address));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::derive_datatype");

      blocklengths[num_arguments-sizeof...(Members)-1] = 1;
      displacements[num_arguments-sizeof...(Members)-1] = member_address-value_address;
      mpi_datatypes[num_arguments-sizeof...(Members)-1]
        = ::yampi::datatype_of<Member>::call().mpi_datatype();

      ::yampi::derive_datatype_detail::derive_datatype(
        blocklengths, displacements, mpi_datatypes, value, value_address, members...);
    }

    template <
      std::size_t num_arguments, typename Value,
      typename T, std::size_t blocklength, typename... Members>
    inline void derive_datatype(
      YAMPI_array<int, num_arguments>& blocklengths,
      YAMPI_array<MPI_Aint, num_arguments>& displacements,
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,
      Value const& value, MPI_Aint const& value_address,
      T const (&array)[blocklength], Members const&... members)
    {
      MPI_Aint member_address;
      int const error_code
        = MPI_Get_address(array, YAMPI_addressof(member_address));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::derive_datatype");

      blocklengths[num_arguments-sizeof...(Members)-1] = blocklength;
      displacements[num_arguments-sizeof...(Members)-1] = member_address-value_address;
      mpi_datatypes[num_arguments-sizeof...(Members)-1]
        = ::yampi::datatype_of<T>::call().mpi_datatype();

      ::yampi::derive_datatype_detail::derive_datatype(
        blocklengths, displacements, mpi_datatypes, value, value_address, members...);
    }
  }

  template <typename Value, typename... Members>
  inline void derive_datatype(Value const& value, Members const&... members)
  {
    MPI_Aint value_address;
    auto const error_code
      = MPI_Get_address(YAMPI_addressof(value), YAMPI_addressof(value_address));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::derive_datatype");

    YAMPI_array<int, sizeof...(Members)> blocklengths;
    YAMPI_array<MPI_Aint, sizeof...(Members)> displacements;
    YAMPI_array<MPI_Datatype, sizeof...(Members)> mpi_datatypes;

    ::yampi::derive_datatype_detail::derive_datatype(
      blocklengths, displacements, mpi_datatypes, value, value_address, members...);
  }
# else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
  //
  //
  // TODO: implement here using Boost.Preprocessor
  //
  //
  namespace derive_datatype_detail
  {
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#     ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE_NONARRAY(z, n, _) \
    template <std::size_t num_arguments, typename Value, typename Member BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      Member const& member BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      auto member_address = MPI_Aint{};\
      auto const error_code\
        = MPI_Get_address(YAMPI_addressof(member), YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
      blocklengths[num_arguments-n-1] = 1;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<Member>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#       define YAMPI_DERIVE_DATATYPE_ARRAY(z, n, _) \
    template <\
      std::size_t num_arguments, typename Value, typename T, std::size_t blocklength BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      T const (&array)[blocklength] BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      auto member_address = MPI_Aint{};\
      auto const error_code = MPI_Get_address(array, YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
      blocklengths[num_arguments-n-1] = blocklength;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<T>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#     else // BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE_NONARRAY(z, n, _) \
    template <std::size_t num_arguments, typename Value, typename Member BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      Member const& member BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      MPI_Aint member_address{};\
      int const error_code\
        = MPI_Get_address(YAMPI_addressof(member), YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
      blocklengths[num_arguments-n-1] = 1;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<Member>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#       define YAMPI_DERIVE_DATATYPE_ARRAY(z, n, _) \
    template <\
      std::size_t num_arguments, typename Value, typename T, std::size_t blocklength BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      T const (&array)[blocklength] BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      MPI_Aint member_address{};\
      int const error_code = MPI_Get_address(array, YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
      blocklengths[num_arguments-n-1] = blocklength;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<T>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#     endif // BOOST_NO_CXX11_AUTO_DECLARATIONS
#   else // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#     ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE_NONARRAY(z, n, _) \
    template <std::size_t num_arguments, typename Value, typename Member BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      Member const& member BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      auto member_address = MPI_Aint();\
      auto const error_code\
        = MPI_Get_address(YAMPI_addressof(member), YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
      blocklengths[num_arguments-n-1] = 1;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<Member>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#       define YAMPI_DERIVE_DATATYPE_ARRAY(z, n, _) \
    template <\
      std::size_t num_arguments, typename Value, typename T, std::size_t blocklength BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      T const (&array)[blocklength] BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      auto member_address = MPI_Aint();\
      auto const error_code = MPI_Get_address(array, YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
      blocklengths[num_arguments-n-1] = blocklength;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<T>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#     else // BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE_NONARRAY(z, n, _) \
    template <std::size_t num_arguments, typename Value, typename Member BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      Member const& member BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      MPI_Aint member_address;\
      int const error_code\
        = MPI_Get_address(YAMPI_addressof(member), YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
      blocklengths[num_arguments-n-1] = 1;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<Member>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#       define YAMPI_DERIVE_DATATYPE_ARRAY(z, n, _) \
    template <\
      std::size_t num_arguments, typename Value, typename T, std::size_t blocklength BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
    inline void derive_datatype(\
      YAMPI_array<int, num_arguments>& blocklengths,\
      YAMPI_array<MPI_Aint, num_arguments>& displacements,\
      YAMPI_array<MPI_Datatype, num_arguments>& mpi_datatypes,\
      Value const& value, MPI_Aint const& value_address,\
      T const (&array)[blocklength] BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
    {\
      MPI_Aint member_address;\
      int const error_code = MPI_Get_address(array, YAMPI_addressof(member_address));\
\
      if (error_code != MPI_SUCCESS)\
        throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
      blocklengths[num_arguments-n-1] = blocklength;\
      displacements[num_arguments-n-1] = member_address-value_address;\
      mpi_datatypes[num_arguments-n-1]\
        = ::yampi::datatype_of<T>::call().mpi_datatype();\
\
      ::yampi::derive_datatype_detail::derive_datatype(\
        blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
    }\

#     endif // BOOST_NO_CXX11_AUTO_DECLARATIONS
#   endif // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX

    BOOST_PP_REPEAT(BOOST_PP_DEC(YAMPI_NUM_DERIVE_DATATYPE_MEMBERS), YAMPI_DERIVE_DATATYPE_NONARRAY, _)
    BOOST_PP_REPEAT(BOOST_PP_DEC(YAMPI_NUM_DERIVE_DATATYPE_MEMBERS), YAMPI_DERIVE_DATATYPE_ARRAY, _)

#   undef YAMPI_DERIVE_DATATYPE_NONARRAY
#   undef YAMPI_DERIVE_DATATYPE_ARRAY
  }

#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#     ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE(z, n, _) \
  template <typename Value BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
  inline void derive_datatype(Value const& value BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
  {\
    auto value_address = MPI_Aint{};\
    auto const error_code = MPI_Get_address(YAMPI_addressof(value), YAMPI_addressof(value_address));\
\
    if (error_code != MPI_SUCCESS)\
      throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
    YAMPI_array<int, n> blocklengths;\
    YAMPI_array<MPI_Aint, n> displacements;\
    YAMPI_array<MPI_Datatype, n> mpi_datatypes;\
\
    ::yampi::derive_datatype_detail::derive_datatype(\
      blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
  }\

#     else // BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE(z, n, _) \
  template <typename Value BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
  inline void derive_datatype(Value const& value BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
  {\
    MPI_Aint value_address{};\
    int const error_code = MPI_Get_address(YAMPI_addressof(value), YAMPI_addressof(value_address));\
\
    if (error_code != MPI_SUCCESS)\
      throw ::yampi::error{error_code, "yampi::derive_datatype"};\
\
    YAMPI_array<int, n> blocklengths;\
    YAMPI_array<MPI_Aint, n> displacements;\
    YAMPI_array<MPI_Datatype, n> mpi_datatypes;\
\
    ::yampi::derive_datatype_detail::derive_datatype(\
      blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
  }\

#     endif // BOOST_NO_CXX11_AUTO_DECLARATIONS
#   else // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#     ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE(z, n, _) \
  template <typename Value BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
  inline void derive_datatype(Value const& value BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
  {\
    auto value_address = MPI_Aint();\
    auto const error_code = MPI_Get_address(YAMPI_addressof(value), YAMPI_addressof(value_address));\
\
    if (error_code != MPI_SUCCESS)\
      throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
    YAMPI_array<int, n> blocklengths;\
    YAMPI_array<MPI_Aint, n> displacements;\
    YAMPI_array<MPI_Datatype, n> mpi_datatypes;\
\
    ::yampi::derive_datatype_detail::derive_datatype(\
      blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
  }\

#     else // BOOST_NO_CXX11_AUTO_DECLARATIONS
#       define YAMPI_DERIVE_DATATYPE(z, n, _) \
  template <typename Value BOOST_PP_ENUM_TRAILING_PARAMS(n, typename Members)>\
  inline void derive_datatype(Value const& value BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, Members, const& members))\
  {\
    MPI_Aint value_address;\
    int const error_code = MPI_Get_address(YAMPI_addressof(value), YAMPI_addressof(value_address));\
\
    if (error_code != MPI_SUCCESS)\
      throw ::yampi::error(error_code, "yampi::derive_datatype");\
\
    YAMPI_array<int, n> blocklengths;\
    YAMPI_array<MPI_Aint, n> displacements;\
    YAMPI_array<MPI_Datatype, n> mpi_datatypes;\
\
    ::yampi::derive_datatype_detail::derive_datatype(\
      blocklengths, displacements, mpi_datatypes, value, value_address BOOST_PP_ENUM_TRAILING_PARAMS(n, members));\
  }\

#     endif // BOOST_NO_CXX11_AUTO_DECLARATIONS
#   endif // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX

  BOOST_PP_REPEAT(YAMPI_NUM_DERIVE_DATATYPE_MEMBERS, YAMPI_DERIVE_DATATYPE, _)

#   undef YAMPI_DERIVE_DATATYPE
# endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES


  namespace dispatch
  {
    template <typename Value>
    struct derive_datatype
    {
      static void call(Value const& value)
      { ::yampi::access::derive_datatype(value); }
    };
  }


  template <typename Value>
  inline void derive_datatype()
  {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    ::yampi::dispatch::derive_datatype<Value>::call(Value{});
# else
    ::yampi::dispatch::derive_datatype<Value>::call(Value());
# endif
  }
}


# undef YAMPI_array
# undef YAMPI_addressof

#endif

