#ifndef YAMPI_UNCOMMITTED_DATATYPE_HPP
# define YAMPI_UNCOMMITTED_DATATYPE_HPP

# include <boost/config.hpp>

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
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
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
# include <yampi/address.hpp>
# include <yampi/bounds.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   define YAMPI_is_convertible std::is_convertible
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_is_same boost::is_same
#   define YAMPI_is_convertible boost::is_convertible
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
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
    ::yampi::address stride_bytes_;

   public:
    BOOST_CONSTEXPR heterogeneous_strided_block(int const length, ::yampi::address const& stride_bytes)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::address >::value)
      : length_(length), stride_bytes_(stride_bytes)
    { }

    BOOST_CONSTEXPR int const& length() const BOOST_NOEXCEPT_OR_NOTHROW { return length_; }
    BOOST_CONSTEXPR ::yampi::address const& stride_bytes() const BOOST_NOEXCEPT_OR_NOTHROW { return stride_bytes_; }

    void swap(heterogeneous_strided_block& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_swappable< ::yampi::address >::value)
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


  class datatype;

  class uncommitted_datatype
  {
    MPI_Datatype mpi_datatype_;
    bool is_committed_;
    friend class ::yampi::datatype;

   public:
# if MPI_VERSION >= 3
    typedef MPI_Count size_type;
    typedef MPI_Count count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# else
    typedef int size_type;
    typedef MPI_Aint count_type;
    typedef ::yampi::bounds<count_type> bounds_type;
# endif

    uncommitted_datatype()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(MPI_DATATYPE_NULL), is_committed_(false)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    uncommitted_datatype(uncommitted_datatype const&) = default;
    uncommitted_datatype& operator=(uncommitted_datatype const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    uncommitted_datatype(uncommitted_datatype&&) = default;
    uncommitted_datatype& operator=(uncommitted_datatype&&) = default;
#   endif
    ~uncommitted_datatype() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    uncommitted_datatype(MPI_Datatype const& mpi_datatype, bool const is_committed)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_(mpi_datatype), is_committed_(is_committed)
    { }

    uncommitted_datatype(
      uncommitted_datatype const& old_datatype,
      ::yampi::environment const& environment)
      : mpi_datatype_(duplicate(old_datatype, environment)), is_committed_(false)
    { }

    // MPI_Type_contiguous
    uncommitted_datatype(
      uncommitted_datatype const& base_datatype, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, count, environment)), is_committed_(false)
    { }

    // MPI_Type_vector
    uncommitted_datatype(
      uncommitted_datatype const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, block, count, environment)), is_committed_(false)
    { }

    // MPI_Type_create_hvector
    uncommitted_datatype(
      uncommitted_datatype const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, int const count,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(base_datatype, block, count, environment)), is_committed_(false)
    { }

    // MPI_Type_indexed, MPI_Type_create_hindexed
    template <typename ContiguousIterator1, typename ContiguousIterator2>
    uncommitted_datatype(
      uncommitted_datatype const& base_datatype,
      ContiguousIterator1 const displacement_first,
      ContiguousIterator1 const displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            base_datatype, displacement_first, displacement_last, block_length_first,
            environment)),
        is_committed_(false)
    { }

    // MPI_Type_create_indexed_block, MPI_Type_create_hindexed_block
    template <typename ContiguousIterator>
    uncommitted_datatype(
      uncommitted_datatype const& base_datatype,
      ContiguousIterator const displacement_first,
      ContiguousIterator const displacement_last,
      int const block_length,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            base_datatype, displacement_first, displacement_last, block_length,
            environment)),
        is_committed_(false)
    { }

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    uncommitted_datatype(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_(
          derive(
            datatype_first, datatype_last, byte_displacement_first, block_length_first,
            environment)),
        is_committed_(false)
    { }

    // MPI_Type_create_subarray
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    uncommitted_datatype(
      uncommitted_datatype const& array_element_datatype,
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
            environment)),
        is_committed_(false)
    { }

    uncommitted_datatype(
      uncommitted_datatype const& old_datatype,
      ::yampi::bounds<count_type> const& new_bounds,
      ::yampi::environment const& environment)
      : mpi_datatype_(derive(old_datatype, new_bounds, environment)), is_committed_(false)
    { }

   private:
    MPI_Datatype duplicate(
      uncommitted_datatype const& old_datatype,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_dup(old_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::duplicate", environment);
    }

    // MPI_Type_contiguous
    MPI_Datatype derive(
      uncommitted_datatype const& base_datatype, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_contiguous(
            count, base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_vector
    MPI_Datatype derive(
      uncommitted_datatype const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_vector(
            count, block.length(), block.stride(),
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_create_hvector
    MPI_Datatype derive(
      uncommitted_datatype const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, int const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_hvector(
            count, block.length(), block.stride_bytes().mpi_address(),
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_indexed
    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      not YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator1>::value_type,
        ::yampi::address>::value,
      MPI_Datatype>::type
    derive(
      uncommitted_datatype const& base_datatype,
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
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_create_hindexed
    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator1>::value_type,
        ::yampi::address>::value,
      MPI_Datatype>::type
    derive(
      uncommitted_datatype const& base_datatype,
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
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_create_indexed_block
    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      not YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator>::value_type,
        ::yampi::address>::value,
      MPI_Datatype>::type
    derive(
      uncommitted_datatype const& base_datatype,
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
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_create_hindexed_block
    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      YAMPI_is_same<
        typename std::iterator_traits<ContiguousIterator>::value_type,
        ::yampi::address>::value,
      MPI_Datatype>::type
    derive(
      uncommitted_datatype const& base_datatype,
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
            base_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
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
           uncommitted_datatype const>::value),
        "The value type of ContiguousIterator1 must be convertible to"
        " yampi::uncommitted_datatype const");
      static_assert(
        (YAMPI_is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           ::yampi::address const>::value),
        "The value type of ContiguousIterator2 must be convertible to"
        " yampi::address const");
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
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    // MPI_Type_create_subarray
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    MPI_Datatype derive(
      uncommitted_datatype const& array_element_datatype,
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
            MPI_ORDER_C, array_element_datatype.mpi_datatype_, YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

    MPI_Datatype derive(
      uncommitted_datatype const& old_datatype,
      ::yampi::bounds<count_type> const& new_bounds,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_resized(
            old_datatype.mpi_datatype_,
            static_cast<MPI_Aint>(new_bounds.lower_bound()),
            static_cast<MPI_Aint>(new_bounds.extent()),
            YAMPI_addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::derive", environment);
    }

   public:
    bool operator==(uncommitted_datatype const& other) const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_datatype_ == other.mpi_datatype_))
    { return mpi_datatype_ == other.mpi_datatype_; }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_datatype_ == MPI_DATATYPE_NULL))
    { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    size_type size(::yampi::environment const& environment) const
    {
      size_type result;
      int const error_code = YAMPI_Type_size(mpi_datatype_, YAMPI_addressof(result));
      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(
            error_code, "yampi::uncommitted_datatype::size", environment);
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
            error_code, "yampi::uncommitted_datatype::bounds", environment);
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
            error_code, "yampi::uncommitted_datatype::true_bounds", environment);
    }

    MPI_Datatype const& mpi_datatype() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_datatype_; }

    void swap(uncommitted_datatype& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(
    ::yampi::uncommitted_datatype const& lhs, ::yampi::uncommitted_datatype const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::uncommitted_datatype& lhs, ::yampi::uncommitted_datatype& rhs)
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
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_enable_if
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_convertible
# undef YAMPI_is_same


#endif

