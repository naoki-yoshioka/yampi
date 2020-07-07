#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <cassert>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/is_convertible.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
#   include <boost/utility/enable_if.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/bounds.hpp>
# include <yampi/datatype_base.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   define YAMPI_is_convertible std::is_convertible
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_is_same boost::is_same
#   define YAMPI_is_convertible boost::is_convertible
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
#   define YAMPI_enable_if boost::enable_if_c
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif

# if MPI_VERSION >= 3
#   define YAMPI_Type_size MPI_Type_size_x
#   define YAMPI_Type_get_extent MPI_Type_get_extent_x
#   define YAMPI_Type_get_true_extent MPI_Type_get_true_extent_x
# else
#   define YAMPI_Type_size MPI_Type_size
#   define YAMPI_Type_get_extent MPI_Type_get_extent
#   define YAMPI_Type_get_true_extent MPI_Type_get_true_extent
# endif


namespace yampi
{
  class strided_block
  {
    int length_;
    int stride_;

   public:
    BOOST_CONSTEXPR strided_block(int const length, int const stride) BOOST_NOEXCEPT_OR_NOTHROW
      : length_(length), stride_(stride)
    { }

    BOOST_CONSTEXPR int const& length() const BOOST_NOEXCEPT_OR_NOTHROW { return length_; }
    BOOST_CONSTEXPR int const& stride() const BOOST_NOEXCEPT_OR_NOTHROW { return stride_; }

    void swap(strided_block& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_, other.stride_);
    }
  };

  BOOST_CONSTEXPR inline bool operator==(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return lhs.length() == rhs.length() and lhs.stride() == rhs.stride(); }

  BOOST_CONSTEXPR inline bool operator!=(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline void swap(::yampi::strided_block& lhs, ::yampi::strided_block& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs.swap(rhs); }


  class heterogeneous_strided_block
  {
    int length_;
    ::yampi::byte_displacement stride_bytes_;

   public:
    BOOST_CONSTEXPR heterogeneous_strided_block(int const length, ::yampi::byte_displacement const& stride_bytes)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::byte_displacement >::value)
      : length_(length), stride_bytes_(stride_bytes)
    { }

    BOOST_CONSTEXPR int const& length() const BOOST_NOEXCEPT_OR_NOTHROW { return length_; }
    BOOST_CONSTEXPR ::yampi::byte_displacement const& stride_bytes() const BOOST_NOEXCEPT_OR_NOTHROW { return stride_bytes_; }

    void swap(heterogeneous_strided_block& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_swappable< ::yampi::byte_displacement >::value)
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_bytes_, other.stride_bytes_);
    }
  };

  BOOST_CONSTEXPR inline bool operator==(
    ::yampi::heterogeneous_strided_block const& lhs,
    ::yampi::heterogeneous_strided_block const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.stride_bytes() == rhs.stride_bytes()))
  {
    return
      lhs.length() == rhs.length() and lhs.stride_bytes() == rhs.stride_bytes();
  }

  BOOST_CONSTEXPR inline bool operator!=(
    ::yampi::heterogeneous_strided_block const& lhs,
    ::yampi::heterogeneous_strided_block const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::heterogeneous_strided_block& lhs,
    ::yampi::heterogeneous_strided_block& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  namespace datatype_detail
  {
    inline bool is_predefined_mpi_datatype(MPI_Datatype const& mpi_datatype)
    {
      return
        mpi_datatype == MPI_CHAR or mpi_datatype == MPI_SHORT
        or mpi_datatype == MPI_INT or mpi_datatype == MPI_LONG
#   ifndef BOOST_NO_LONG_LONG
        or mpi_datatype == MPI_LONG_LONG
#   endif
        or mpi_datatype == MPI_SIGNED_CHAR
        or mpi_datatype == MPI_UNSIGNED_CHAR or mpi_datatype == MPI_UNSIGNED_SHORT
        or mpi_datatype == MPI_UNSIGNED or mpi_datatype == MPI_UNSIGNED_LONG
#   ifndef BOOST_NO_LONG_LONG
        or mpi_datatype == MPI_UNSIGNED_LONG_LONG
#   endif
        or mpi_datatype == MPI_FLOAT or mpi_datatype == MPI_DOUBLE or mpi_datatype == MPI_LONG_DOUBLE
        or mpi_datatype == MPI_WCHAR
#   if MPI_VERSION >= 3
        or mpi_datatype == MPI_CXX_BOOL or mpi_datatype == MPI_CXX_FLOAT_COMPLEX
        or mpi_datatype == MPI_CXX_DOUBLE_COMPLEX or mpi_datatype == MPI_CXX_LONG_DOUBLE_COMPLEX
#   elif MPI_VERSION >= 2
        or mpi_datatype == MPI::BOOL or mpi_datatype == MPI::COMPLEX
        or mpi_datatype == MPI::DOUBLE_COMPLEX or mpi_datatype == MPI::LONG_DOUBLE_COMPLEX
#   endif
        or mpi_datatype == MPI_SHORT_INT or mpi_datatype == MPI_2INT or mpi_datatype == MPI_LONG_INT
        or mpi_datatype == MPI_FLOAT_INT or mpi_datatype == MPI_DOUBLE_INT or mpi_datatype == MPI_LONG_DOUBLE_INT;
    }
  }

  class datatype
    : public ::yampi::datatype_base< ::yampi::datatype >
  {
    typedef ::yampi::datatype_base< ::yampi::datatype > super_type;

    MPI_Datatype mpi_datatype_;

   public:
    datatype()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    datatype(datatype const&) = delete;
    datatype& operator=(datatype const&) = delete;
# else
   private:
    datatype(datatype const&);
    datatype& operator=(datatype const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Datatype>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Datatype>::value)
      : mpi_datatype_(std::move(other.mpi_datatype_))
    { other.mpi_datatype_ = MPI_DATATYPE_NULL; }

    datatype& operator=(datatype&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Datatype>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Datatype>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_datatype_ = std::move(other.mpi_datatype_);
        other.mpi_datatype_ = MPI_DATATYPE_NULL;
      }
      return *this;
    }
# endif

    ~datatype() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL
          or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      MPI_Type_free(YAMPI_addressof(mpi_datatype_));
    }

    explicit datatype(MPI_Datatype const& mpi_datatype)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(mpi_datatype)
    { }

    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::environment const& environment)
      : mpi_datatype_(duplicate(base_datatype, environment))
    { }

    // MPI_Type_contiguous
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, count, environment))
    { }

    // MPI_Type_vector
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, block, count, environment))
    { }

    // MPI_Type_create_hvector
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, block, count, environment))
    { }

    // MPI_Type_indexed, MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator1 const displacement_first,
      ContiguousIterator1 const displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            base_datatype, displacement_first, displacement_last, block_length_first,
            environment))
    { }

    // MPI_Type_create_indexed_block, MPI_Type_create_hindexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator const displacement_first,
      ContiguousIterator const displacement_last,
      int const block_length,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            base_datatype, displacement_first, displacement_last, block_length,
            environment))
    { }

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    datatype(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            datatype_first, datatype_last, byte_displacement_first, block_length_first,
            environment))
    { }

    // MPI_Type_create_subarray
    template <
      typename DerivedDatatype,
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& array_element_datatype,
      ContiguousIterator1 const array_size_first,
      ContiguousIterator1 const array_size_last,
      ContiguousIterator2 const array_subsize_first,
      ContiguousIterator3 const array_start_index_first,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            array_element_datatype,
            array_size_first, array_size_last,
            array_subsize_first, array_start_index_first,
            environment))
    { }

    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& old_datatype,
      ::yampi::bounds<count_type> const& new_bounds,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(old_datatype, new_bounds, environment))
    { }

   private:
    template <typename DerivedDatatype>
    MPI_Datatype duplicate(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_dup(base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::datatype::duplicate", environment);
    }

    // MPI_Type_contiguous
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_contiguous(
            count, base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_vector
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_vector(
            count, block.length(), block.stride(),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hvector
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_hvector(
            count, block.length(), block.stride_bytes().mpi_byte_displacement(),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_indexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      not YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator1>::value_type,
        ::yampi::byte_displacement>::value,
      MPI_Datatype>::type
    derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator1 const displacement_first,
      ContiguousIterator1 const displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           int const>::value),
        "The value type of ContiguousIterator1 must be convertible to int const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_indexed(
            displacement_last - displacement_first,
            YAMPI_addressof(*block_length_first),
            YAMPI_addressof(*displacement_first),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator1>::value_type,
        ::yampi::byte_displacement>::value,
      MPI_Datatype>::type
    derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator1 const byte_displacement_first,
      ContiguousIterator1 const byte_displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_hindexed(
            byte_displacement_last - byte_displacement_first,
            YAMPI_addressof(*block_length_first),
            reinterpret_cast<MPI_Aint const*>(
              YAMPI_addressof(*byte_displacement_first)),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_indexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    typename YAMPI_enable_if<
      not YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator>::value_type,
        ::yampi::byte_displacement>::value,
      MPI_Datatype>::type
    derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator const displacement_first,
      ContiguousIterator const displacement_last,
      int const block_length,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           int const>::value),
        "The value type of ContiguousIterator must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_indexed_block(
            displacement_last - displacement_first,
            block_length, YAMPI_addressof(*displacement_first),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hindexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    typename YAMPI_enable_if<
      YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator>::value_type,
        ::yampi::byte_displacement>::value,
      MPI_Datatype>::type
    derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator const byte_displacement_first,
      ContiguousIterator const byte_displacement_last,
      int const block_length,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_hindexed_block(
            byte_displacement_last - byte_displacement_first,
            block_length,
            reinterpret_cast<MPI_Aint const*>(
              YAMPI_addressof(*byte_displacement_first)),
            base_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    MPI_Datatype derive(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           ::yampi::datatype const>::value),
        "The value type of ContiguousIterator1 must be convertible to"
        " yampi::datatype const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           ::yampi::byte_displacement const>::value),
        "The value type of ContiguousIterator2 must be convertible to"
        " yampi::byte_displacement const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator3>::value_type,
           int const>::value),
        "The value type of ContiguousIterator3 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_struct(
            datatype_last - datatype_first,
            YAMPI_addressof(*block_length_first),
            reinterpret_cast<MPI_Aint const*>(
              YAMPI_addressof(*byte_displacement_first)),
            reinterpret_cast<MPI_Datatype const*>(YAMPI_addressof(*datatype_first)),
            YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_subarray
    template <
      typename DerivedDatatype,
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& array_element_datatype,
      ContiguousIterator1 const array_size_first,
      ContiguousIterator1 const array_size_last,
      ContiguousIterator2 const array_subsize_first,
      ContiguousIterator3 const array_start_index_first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           int const>::value),
        "The value type of ContiguousIterator1 must be convertible to int const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator3>::value_type,
           int const>::value),
        "The value type of ContiguousIterator3 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_subarray(
            array_size_last - array_size_first,
            YAMPI_addressof(*array_size_first),
            YAMPI_addressof(*array_subsize_first),
            YAMPI_addressof(*array_start_index_first),
            MPI_ORDER_C, array_element_datatype.mpi_datatype(), YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& old_datatype,
      ::yampi::bounds<count_type> const& new_bounds,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_resized(
            old_datatype.mpi_datatype(),
            static_cast<MPI_Aint>(new_bounds.lower_bound()),
            static_cast<MPI_Aint>(new_bounds.extent()),
            YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    MPI_Datatype commit(
      MPI_Datatype const& mpi_datatype,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result = mpi_datatype;
      int const error_code = MPI_Type_commit(YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::datatype::commit", environment);
    }

   public:
    bool operator==(datatype const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_datatype_ == other.mpi_datatype_; }

    bool do_is_null() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    MPI_Datatype do_mpi_datatype() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_datatype_; }


    void free(::yampi::environment const& environment)
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL
          or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      int const error_code = MPI_Type_free(YAMPI_addressof(mpi_datatype_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::free", environment);
    }


    void reset(
      MPI_Datatype const& mpi_datatype, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = mpi_datatype;
    }

    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = duplicate(base_datatype, environment);
    }

    // MPI_Type_contiguous
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype, int const count,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, count, environment);
    }

    // MPI_Type_vector
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, block, count, environment);
    }

    // MPI_Type_create_hvector
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, int const count,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, block, count, environment);
    }

    // MPI_Type_indexed, MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator1 const displacement_first,
      ContiguousIterator1 const displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment)
    {
      free(environment);

      mpi_datatype_
        = derive(
            base_datatype,
            displacement_first, displacement_last, block_length_first,
            environment);
    }

    // MPI_Type_create_indexed_block, MPI_Type_create_hindexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator const displacement_first,
      ContiguousIterator const displacement_last,
      int const block_length,
      ::yampi::environment const& environment)
    {
      free(environment);

      mpi_datatype_
        = derive(
            base_datatype,
            displacement_first, displacement_last, block_length,
            environment);
    }

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    void reset(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment)
    {
      free(environment);

      mpi_datatype_
        = derive(
            datatype_first, datatype_last, byte_displacement_first, block_length_first,
            environment);
    }

    // MPI_Type_create_subarray
    template <
      typename DerivedDatatype,
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& array_element_datatype,
      ContiguousIterator1 const array_size_first,
      ContiguousIterator1 const array_size_last,
      ContiguousIterator2 const array_subsize_first,
      ContiguousIterator3 const array_start_index_first,
      ::yampi::environment const& environment)
    {
      free(environment);

      mpi_datatype_
        = derive(
            array_element_datatype,
            array_size_first, array_size_last,
            array_subsize_first, array_start_index_first,
            environment);
    }

    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& old_datatype,
      ::yampi::bounds<count_type> const& new_bounds,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(old_datatype, new_bounds, environment);
    }


    size_type size(::yampi::environment const& environment) const
    {
      size_type result;
      int const error_code = YAMPI_Type_size(mpi_datatype_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::datatype::size", environment);
    }

    bounds_type bounds(::yampi::environment const& environment) const
    {
      count_type lower_bound, extent;
      int const error_code
        = YAMPI_Type_get_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      return error_code == MPI_SUCCESS
        ? ::yampi::make_bounds(lower_bound, extent)
        : throw ::yampi::error(
            error_code, "yampi::datatype::bounds", environment);
    }

    bounds_type true_bounds(::yampi::environment const& environment) const
    {
      count_type lower_bound, extent;
      int const error_code
        = YAMPI_Type_get_true_extent(
            mpi_datatype_, YAMPI_addressof(lower_bound), YAMPI_addressof(extent));
      return error_code == MPI_SUCCESS
        ? ::yampi::make_bounds(lower_bound, extent)
        : throw ::yampi::error(
            error_code, "yampi::datatype::true_bounds", environment);
    }

    void swap(datatype& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_Type_get_true_extent
# undef YAMPI_Type_get_extent
# undef YAMPI_Type_size
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_enable_if
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_convertible
# undef YAMPI_is_same

#endif

