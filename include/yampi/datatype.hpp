#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/byte_displacement.hpp>
# include <yampi/count.hpp>
# include <yampi/bounds.hpp>
# include <yampi/extent.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/predefined_datatype.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class strided_block
  {
    ::yampi::count length_;
    ::yampi::count stride_;

   public:
    constexpr strided_block(::yampi::count const length, ::yampi::count const stride)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : length_{length}, stride_{stride}
    { }

    constexpr ::yampi::count const& length() const noexcept { return length_; }
    constexpr ::yampi::count const& stride() const noexcept { return stride_; }

    void swap(strided_block& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_, other.stride_);
    }
  };

  inline constexpr bool operator==(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs) noexcept
  { return lhs.length() == rhs.length() and lhs.stride() == rhs.stride(); }

  inline constexpr bool operator!=(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::strided_block& lhs, ::yampi::strided_block& rhs) noexcept
  { lhs.swap(rhs); }


  class heterogeneous_strided_block
  {
    ::yampi::count length_;
# if MPI_VERSION >= 4
    ::yampi::count stride_bytes_;
# else // MPI_VERSION >= 4
    ::yampi::byte_displacement stride_bytes_;
# endif // MPI_VERSION >= 4

   public:
# if MPI_VERSION >= 4
    constexpr heterogeneous_strided_block(::yampi::count const length, ::yampi::count const& stride_bytes)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : length_{length}, stride_bytes_{stride_bytes}
    { }

    constexpr heterogeneous_strided_block(::yampi::count const length, ::yampi::byte_displacement const& stride_bytes)
      noexcept(
        std::is_nothrow_copy_constructible< ::yampi::count >::value
        and std::is_nothrow_copy_constructible< ::yampi::byte_displacement >::value)
      : length_{length}, stride_bytes_{static_cast<MPI_Count>(stride_bytes.mpi_byte_displacement())}
    { }
# else // MPI_VERSION >= 4
    constexpr heterogeneous_strided_block(::yampi::count const length, ::yampi::count const& stride_bytes)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : length_{length}, stride_bytes_{static_cast<MPI_Aint>(stride_bytes.mpi_count())}
    { }

    constexpr heterogeneous_strided_block(::yampi::count const length, ::yampi::byte_displacement const& stride_bytes)
      noexcept(
        std::is_nothrow_copy_constructible< ::yampi::count >::value
        and std::is_nothrow_copy_constructible< ::yampi::byte_displacement >::value)
      : length_{length}, stride_bytes_{stride_bytes}
    { }
# endif // MPI_VERSION >= 4

    constexpr ::yampi::count const& length() const noexcept { return length_; }
# if MPI_VERSION >= 4
    constexpr ::yampi::count const& stride_bytes() const noexcept { return stride_bytes_; }
# else // MPI_VERSION >= 4
    constexpr ::yampi::byte_displacement const& stride_bytes() const noexcept { return stride_bytes_; }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 4
    void swap(heterogeneous_strided_block& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_bytes_, other.stride_bytes_);
    }
# else // MPI_VERSION >= 4
    void swap(heterogeneous_strided_block& other)
      noexcept(
        YAMPI_is_nothrow_swappable< ::yampi::count >::value
        and YAMPI_is_nothrow_swappable< ::yampi::byte_displacement >::value)
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_bytes_, other.stride_bytes_);
    }
# endif // MPI_VERSION >= 4
  };

  inline constexpr bool operator==(
    ::yampi::heterogeneous_strided_block const& lhs, ::yampi::heterogeneous_strided_block const& rhs)
    noexcept(noexcept(lhs.stride_bytes() == rhs.stride_bytes()))
  { return lhs.length() == rhs.length() and lhs.stride_bytes() == rhs.stride_bytes(); }

  inline constexpr bool operator!=(
    ::yampi::heterogeneous_strided_block const& lhs, ::yampi::heterogeneous_strided_block const& rhs)
    noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::heterogeneous_strided_block& lhs, ::yampi::heterogeneous_strided_block& rhs)
    noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  class flexible_blocks
  {
   public:
# if MPI_VERSION >= 4
    using length_type = ::yampi::count;
    using displacement_type = ::yampi::count;
# else // MPI_VERSION >= 4
    using length_type = int;
    using displacement_type = int;
# endif // MPI_VERSION >= 4

   private:
    ::yampi::count count_;
    length_type* length_first_;
    displacement_type* displacement_first_;

   public:
    template <typename LengthIterator, typename DisplacementIterator>
    constexpr flexible_blocks(
      LengthIterator const length_first, LengthIterator const length_last,
      DisplacementIterator const displacement_first)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : count_{static_cast<int>(length_last - length_first)},
        length_first_{const_cast<length_type*>(std::addressof(*length_first))},
        displacement_first_{const_cast<displacement_type*>(std::addressof(*displacement_first))}
    {
# if MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of LengthIterator must be convertible to ::yampi::count const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          int const>::value,
        "The value type of LengthIterator must be convertible to int const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          int const>::value,
        "The value type of DisplacementIterator must be convertible to int const");
# endif // MPI_VERSION >= 4
    }

    constexpr ::yampi::count const& count() const noexcept { return count_; }

    constexpr length_type const* length_begin() const noexcept { return length_first_; }
    constexpr length_type const* length_end() const noexcept { return length_first_ + count_.mpi_count(); }
    constexpr displacement_type const* displacement_begin() const noexcept { return displacement_first_; }
    constexpr displacement_type const* displacement_end() const noexcept { return displacement_first_ + count_.mpi_count(); }

    constexpr length_type const& length_front() const noexcept { return *length_begin(); }
    constexpr length_type const& length_back() const noexcept { return *(length_end() - 1); } //{ return *std::prev(length_end()); }
    constexpr displacement_type const& displacement_front() const noexcept { return *displacement_begin(); }
    constexpr displacement_type const& displacement_back() const noexcept { return *(displacement_end() - 1); } //{ return *std::prev(displacement_end()); }

    constexpr length_type const& length(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(length_begin() + n); }
    constexpr displacement_type const& displacement(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(displacement_begin() + n); }

    length_type const& length_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::length_at)"}; return *(length_begin() + n); }
    displacement_type const& displacement_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::displacement_at)"}; return *(displacement_begin() + n); }

    void swap(flexible_blocks& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(count_, other.count_);
      swap(length_first_, other.length_first_);
      swap(displacement_first_, other.displacement_first_);
    }
  };

  inline constexpr bool operator==(
    ::yampi::flexible_blocks const& lhs, ::yampi::flexible_blocks const& rhs) noexcept
  {
    return lhs.count() == rhs.count()
      and lhs.length_begin() == rhs.length_begin()
      and lhs.displacement_begin() == rhs.displacement_begin();
  }

  inline constexpr bool operator!=(
    ::yampi::flexible_blocks const& lhs, ::yampi::flexible_blocks const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::flexible_blocks& lhs, ::yampi::flexible_blocks& rhs) noexcept
  { lhs.swap(rhs); }


  class heterogeneous_flexible_blocks
  {
   public:
# if MPI_VERSION >= 4
    using length_type = ::yampi::count;
    using displacement_type = ::yampi::count;
# else // MPI_VERSION >= 4
    using length_type = int;
    using displacement_type = ::yampi::byte_displacement;
# endif // MPI_VERSION >= 4

   private:
    ::yampi::count count_;
    length_type* length_first_;
    displacement_type* displacement_first_;

   public:
    template <typename LengthIterator, typename DisplacementIterator>
    constexpr heterogeneous_flexible_blocks(
      LengthIterator const length_first, LengthIterator const length_last,
      DisplacementIterator const displacement_first)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : count_{static_cast<int>(length_last - length_first)},
        length_first_{const_cast<length_type*>(std::addressof(*length_first))},
        displacement_first_{const_cast<displacement_type*>(std::addressof(*displacement_first))}
    {
# if MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of LengthIterator must be convertible to ::yampi::count const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          int const>::value,
        "The value type of LengthIterator must be convertible to int const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::byte_displacement const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::byte_displacement const");
# endif // MPI_VERSION >= 4
    }

    constexpr ::yampi::count const& count() const noexcept { return count_; }

    constexpr length_type const* length_begin() const noexcept { return length_first_; }
    constexpr length_type const* length_end() const noexcept { return length_first_ + count_.mpi_count(); }
    constexpr displacement_type const* displacement_begin() const noexcept { return displacement_first_; }
    constexpr displacement_type const* displacement_end() const noexcept { return displacement_first_ + count_.mpi_count(); }

    constexpr length_type const& length_front() const noexcept { return *length_begin(); }
    constexpr length_type const& length_back() const noexcept { return *(length_end() - 1); } //{ return *std::prev(length_end()); }
    constexpr displacement_type const& displacement_front() const noexcept { return *displacement_begin(); }
    constexpr displacement_type const& displacement_back() const noexcept { return *(displacement_end() - 1); } //{ return *std::prev(displacement_end()); }

    constexpr length_type const& length(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(length_begin() + n); }
    constexpr displacement_type const& displacement(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(displacement_begin() + n); }

    length_type const& length_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::length_at)"}; return *(length_begin() + n); }
    displacement_type const& displacement_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::displacement_at)"}; return *(displacement_begin() + n); }

    void swap(heterogeneous_flexible_blocks& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(count_, other.count_);
      swap(length_first_, other.length_first_);
      swap(displacement_first_, other.displacement_first_);
    }
  };

  inline constexpr bool operator==(
    ::yampi::heterogeneous_flexible_blocks const& lhs,
    ::yampi::heterogeneous_flexible_blocks const& rhs) noexcept
  {
    return lhs.count() == rhs.count()
      and lhs.length_begin() == rhs.length_begin()
      and lhs.displacement_begin() == rhs.displacement_begin();
  }

  inline constexpr bool operator!=(
    ::yampi::heterogeneous_flexible_blocks const& lhs,
    ::yampi::heterogeneous_flexible_blocks const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::heterogeneous_flexible_blocks& lhs,
    ::yampi::heterogeneous_flexible_blocks& rhs) noexcept
  { lhs.swap(rhs); }


  class fixed_blocks
  {
   public:
# if MPI_VERSION >= 4
    using displacement_type = ::yampi::count;
# else // MPI_VERSION >= 4
    using displacement_type = int;
# endif // MPI_VERSION >= 4

   private:
    ::yampi::count count_;
    ::yampi::count length_;
    displacement_type* displacement_first_;

   public:
    template <typename DisplacementIterator>
    constexpr fixed_blocks(
      ::yampi::count const length,
      DisplacementIterator const displacement_first, DisplacementIterator const displacement_last)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : count_{static_cast<int>(displacement_last - displacement_first)},
        length_{length},
        displacement_first_{const_cast<displacement_type*>(std::addressof(*displacement_first))}
    {
# if MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          int const>::value,
        "The value type of DisplacementIterator must be convertible to int const");
# endif // MPI_VERSION >= 4
    }

    constexpr ::yampi::count const& count() const noexcept { return count_; }
    constexpr ::yampi::count const& length() const noexcept { return length_; }

    constexpr displacement_type const* displacement_begin() const noexcept { return displacement_first_; }
    constexpr displacement_type const* displacement_end() const noexcept { return displacement_first_ + count_.mpi_count(); }

    constexpr displacement_type const& displacement_front() const noexcept { return *displacement_begin(); }
    constexpr displacement_type const& displacement_back() const noexcept { return *(displacement_end() - 1); } //{ return *std::prev(displacement_end()); }

    constexpr displacement_type const& displacement(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(displacement_begin() + n); }

    displacement_type const& displacement_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::displacement_at)"}; return *(displacement_begin() + n); }

    void swap(fixed_blocks& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(count_, other.count_);
      swap(length_, other.length_);
      swap(displacement_first_, other.displacement_first_);
    }
  };

  inline constexpr bool operator==(
    ::yampi::fixed_blocks const& lhs, ::yampi::fixed_blocks const& rhs) noexcept
  {
    return lhs.count() == rhs.count() and lhs.length() == rhs.length()
      and lhs.displacement_begin() == rhs.displacement_begin();
  }

  inline constexpr bool operator!=(
    ::yampi::fixed_blocks const& lhs, ::yampi::fixed_blocks const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::fixed_blocks& lhs, ::yampi::fixed_blocks& rhs) noexcept
  { lhs.swap(rhs); }
# if MPI_VERSION >= 3


  class heterogeneous_fixed_blocks
  {
   public:
# if MPI_VERSION >= 4
    using displacement_type = ::yampi::count;
# else // MPI_VERSION >= 4
    using displacement_type = ::yampi::byte_displacement;
# endif // MPI_VERSION >= 4

   private:
    ::yampi::count count_;
    ::yampi::count length_;
    displacement_type* displacement_first_;

   public:
    template <typename DisplacementIterator>
    constexpr heterogeneous_fixed_blocks(
      ::yampi::count const length,
      DisplacementIterator const displacement_first, DisplacementIterator const displacement_last)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : count_{static_cast<int>(displacement_last - displacement_first)},
        length_{length},
        displacement_first_{const_cast<displacement_type*>(std::addressof(*displacement_first))}
    {
# if MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::byte_displacement const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::byte_displacement const");
# endif // MPI_VERSION >= 4
    }

    constexpr ::yampi::count const& count() const noexcept { return count_; }
    constexpr ::yampi::count const& length() const noexcept { return length_; }

    constexpr displacement_type const* displacement_begin() const noexcept { return displacement_first_; }
    constexpr displacement_type const* displacement_end() const noexcept { return displacement_first_ + count_.mpi_count(); }

    constexpr displacement_type const& displacement_front() const noexcept { return *displacement_begin(); }
    constexpr displacement_type const& displacement_back() const noexcept { return *(displacement_end() - 1); } //{ return *std::prev(displacement_end()); }

    constexpr displacement_type const& displacement(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(displacement_begin() + n); }

    displacement_type const& displacement_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::displacement_at)"}; return *(displacement_begin() + n); }

    void swap(heterogeneous_fixed_blocks& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(count_, other.count_);
      swap(length_, other.length_);
      swap(displacement_first_, other.displacement_first_);
    }
  };

  inline constexpr bool operator==(
    ::yampi::heterogeneous_fixed_blocks const& lhs,
    ::yampi::heterogeneous_fixed_blocks const& rhs) noexcept
  {
    return lhs.count() == rhs.count() and lhs.length() == rhs.length()
      and lhs.displacement_begin() == rhs.displacement_begin();
  }

  inline constexpr bool operator!=(
    ::yampi::heterogeneous_fixed_blocks const& lhs,
    ::yampi::heterogeneous_fixed_blocks const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::heterogeneous_fixed_blocks& lhs,
    ::yampi::heterogeneous_fixed_blocks& rhs) noexcept
  { lhs.swap(rhs); }
# endif // MPI_VERSION >= 3


  template <typename Datatype>
  class heterogeneous_typed_flexible_blocks
  {
   public:
# if MPI_VERSION >= 4
    using length_type = ::yampi::count;
    using displacement_type = ::yampi::count;
# else // MPI_VERSION >= 4
    using length_type = int;
    using displacement_type = ::yampi::byte_displacement;
# endif // MPI_VERSION >= 4

   private:
    ::yampi::count count_;
    length_type* length_first_;
    displacement_type* displacement_first_;
    Datatype* datatype_first_;

   public:
    template <typename LengthIterator, typename DisplacementIterator, typename DatatypeIterator>
    constexpr heterogeneous_typed_flexible_blocks(
      LengthIterator const length_first, LengthIterator const length_last,
      DisplacementIterator const displacement_first, DatatypeIterator const datatype_first)
      noexcept(std::is_nothrow_copy_constructible< ::yampi::count >::value)
      : count_{static_cast<int>(length_last - length_first)},
        length_first_{const_cast<length_type*>(std::addressof(*length_first))},
        displacement_first_{const_cast<displacement_type*>(std::addressof(*displacement_first))},
        datatype_first_{const_cast< Datatype* >(std::addressof(*datatype_first))}
    {
# if MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of LengthIterator must be convertible to ::yampi::count const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::count const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<LengthIterator>::value_type,
          int const>::value,
        "The value type of LengthIterator must be convertible to int const");
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DisplacementIterator>::value_type,
          ::yampi::byte_displacement const>::value,
        "The value type of DisplacementIterator must be convertible to ::yampi::byte_displacement const");
# endif // MPI_VERSION >= 4
      static_assert(
        std::is_convertible<
          typename std::iterator_traits<DatatypeIterator>::value_type,
          Datatype const>::value,
        "The value type of DatatypeIterator must be convertible to Datatype const");
    }

    constexpr ::yampi::count const& count() const noexcept { return count_; }

    constexpr length_type const* length_begin() const noexcept { return length_first_; }
    constexpr length_type const* length_end() const noexcept { return length_first_ + count_.mpi_count(); }
    constexpr displacement_type const* displacement_begin() const noexcept { return displacement_first_; }
    constexpr displacement_type const* displacement_end() const noexcept { return displacement_first_ + count_.mpi_count(); }
    constexpr Datatype const* datatype_begin() const noexcept { return datatype_first_; }
    constexpr Datatype const* datatype_end() const noexcept { return datatype_first_ + count_.mpi_count(); }

    constexpr length_type const& length_front() const noexcept { return *length_begin(); }
    constexpr length_type const& length_back() const noexcept { return *(length_end() - 1); } //{ return *std::prev(length_end()); }
    constexpr displacement_type const& displacement_front() const noexcept { return *displacement_begin(); }
    constexpr displacement_type const& displacement_back() const noexcept { return *(displacement_end() - 1); } //{ return *std::prev(displacement_end()); }
    constexpr Datatype const& datatype_front() const noexcept { return *datatype_begin(); }
    constexpr Datatype const& datatype_back() const noexcept { return *(datatype_end() - 1); } //{ return *std::prev(datatype_end()); }

    constexpr length_type const& length(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(length_begin() + n); }
    constexpr displacement_type const& displacement(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(displacement_begin() + n); }
    constexpr Datatype const& datatype(int const n) const noexcept { /*assert(n >= 0 and n < count_.int_mpi_count());*/ return *(datatype_begin() + n); }

    length_type const& length_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::length_at)"}; return *(length_begin() + n); }
    displacement_type const& displacement_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::displacement_at)"}; return *(displacement_begin() + n); }
    Datatype const& datatype_at(int const n) const { if (n >= 0 and n < count_.int_mpi_count()) throw std::out_of_range{"out of range (flexible_blocks::datatype_at)"}; return *(datatype_begin() + n); }

    void swap(heterogeneous_typed_flexible_blocks& other)
      noexcept(YAMPI_is_nothrow_swappable< ::yampi::count >::value)
    {
      using std::swap;
      swap(count_, other.count_);
      swap(length_first_, other.length_first_);
      swap(displacement_first_, other.displacement_first_);
      swap(datatype_first_, other.datatype_first_);
    }
  };

  template <typename Datatype>
  inline constexpr bool operator==(
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& lhs,
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& rhs) noexcept
  {
    return lhs.count() == rhs.count()
      and lhs.length_begin() == rhs.length_begin()
      and lhs.displacement_begin() == rhs.displacement_begin()
      and lhs.datatype_begin() == rhs.datatype_begin();
  }

  template <typename Datatype>
  inline constexpr bool operator!=(
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& lhs,
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& rhs) noexcept
  { return not (lhs == rhs); }

  template <typename Datatype>
  inline void swap(
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype>& lhs,
    ::yampi::heterogeneous_typed_flexible_blocks<Datatype>& rhs) noexcept
  { lhs.swap(rhs); }


  namespace datatype_detail
  {
    inline bool is_predefined_mpi_datatype(MPI_Datatype const& mpi_datatype)
    {
      return
        mpi_datatype == MPI_CHAR or mpi_datatype == MPI_SHORT
        or mpi_datatype == MPI_INT or mpi_datatype == MPI_LONG
        or mpi_datatype == MPI_LONG_LONG
        or mpi_datatype == MPI_SIGNED_CHAR
        or mpi_datatype == MPI_UNSIGNED_CHAR or mpi_datatype == MPI_UNSIGNED_SHORT
        or mpi_datatype == MPI_UNSIGNED or mpi_datatype == MPI_UNSIGNED_LONG
        or mpi_datatype == MPI_UNSIGNED_LONG_LONG
        or mpi_datatype == MPI_FLOAT or mpi_datatype == MPI_DOUBLE or mpi_datatype == MPI_LONG_DOUBLE
        or mpi_datatype == MPI_WCHAR
        or mpi_datatype == MPI_AINT or mpi_datatype == MPI_OFFSET
# if MPI_VERSION >= 3
        or mpi_datatype == MPI_COUNT
# endif
# if MPI_VERSION >= 3
        or mpi_datatype == MPI_CXX_BOOL or mpi_datatype == MPI_CXX_FLOAT_COMPLEX
        or mpi_datatype == MPI_CXX_DOUBLE_COMPLEX or mpi_datatype == MPI_CXX_LONG_DOUBLE_COMPLEX
# elif MPI_VERSION >= 2
        or mpi_datatype == MPI::BOOL or mpi_datatype == MPI::COMPLEX
        or mpi_datatype == MPI::DOUBLE_COMPLEX or mpi_datatype == MPI::LONG_DOUBLE_COMPLEX
# endif
        or mpi_datatype == MPI_SHORT_INT or mpi_datatype == MPI_2INT or mpi_datatype == MPI_LONG_INT
        or mpi_datatype == MPI_FLOAT_INT or mpi_datatype == MPI_DOUBLE_INT or mpi_datatype == MPI_LONG_DOUBLE_INT;
    }
  }

  class datatype
    : public ::yampi::datatype_base< ::yampi::datatype >
  {
    typedef ::yampi::datatype_base< ::yampi::datatype > base_type;

    MPI_Datatype mpi_datatype_;

   public:
    datatype() noexcept(std::is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_{MPI_DATATYPE_NULL}
    { }

    datatype(datatype const&) = delete;
    datatype& operator=(datatype const&) = delete;

    datatype(datatype&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Datatype>::value
        and std::is_nothrow_copy_assignable<MPI_Datatype>::value)
      : mpi_datatype_{std::move(other.mpi_datatype_)}
    { other.mpi_datatype_ = MPI_DATATYPE_NULL; }

    datatype& operator=(datatype&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Datatype>::value
        and std::is_nothrow_copy_assignable<MPI_Datatype>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_datatype_ != MPI_DATATYPE_NULL
            and (not ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_)))
          MPI_Type_free(std::addressof(mpi_datatype_));
        mpi_datatype_ = std::move(other.mpi_datatype_);
        other.mpi_datatype_ = MPI_DATATYPE_NULL;
      }
      return *this;
    }

    ~datatype() noexcept
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL
          or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      MPI_Type_free(std::addressof(mpi_datatype_));
    }

    template <typename T>
    explicit datatype(::yampi::predefined_datatype<T> const predefined_datatype)
      : mpi_datatype_{predefined_datatype.mpi_datatype()}
    { }

    explicit datatype(MPI_Datatype const& mpi_datatype)
      noexcept(std::is_nothrow_copy_constructible<MPI_Datatype>::value)
      : mpi_datatype_{mpi_datatype}
    { }

    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::environment const& environment)
      : mpi_datatype_{duplicate(base_datatype, environment)}
    { }

    // MPI_Type_contiguous
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype, ::yampi::count const count,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, count, environment)}
    { }

    // MPI_Type_vector
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::strided_block const& block, ::yampi::count const count,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, block, count, environment)}
    { }

    // MPI_Type_create_hvector
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, ::yampi::count const count,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, block, count, environment)}
    { }

    // MPI_Type_indexed
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::flexible_blocks const& blocks,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, blocks, environment)}
    { }

    // MPI_Type_create_hindexed
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_flexible_blocks const& blocks,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, blocks, environment)}
    { }

    // MPI_Type_create_indexed_block
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::fixed_blocks const& blocks,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, blocks, environment)}
    { }

# if MPI_VERSION >= 3
    // MPI_Type_create_hindexed_block
    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_fixed_blocks const& blocks,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(base_datatype, blocks, environment)}
    { }
# endif // MPI_VERSION >= 3

    // MPI_Type_create_struct
    template <typename Datatype>
    datatype(
      ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& typed_blocks,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(typed_blocks, environment)}
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
      : mpi_datatype_{
          derive(
            array_element_datatype,
            array_size_first, array_size_last,
            array_subsize_first, array_start_index_first,
            environment)}
    { }

    template <typename DerivedDatatype>
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& old_datatype,
      ::yampi::bounds const& new_bounds,
      ::yampi::environment const& environment)
      : mpi_datatype_{derive(old_datatype, new_bounds, environment)}
    { }

    // *** DEPRECATED CONSTRUCTORS *** //
    // MPI_Type_indexed, MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    [[deprecated]]
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator1 const displacement_first,
      ContiguousIterator1 const displacement_last,
      ContiguousIterator2 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_{
          derive(
            base_datatype, displacement_first, displacement_last, block_length_first,
            environment)}
    { }

    // MPI_Type_create_indexed_block, MPI_Type_create_hindexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    [[deprecated]]
    datatype(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ContiguousIterator const displacement_first,
      ContiguousIterator const displacement_last,
      int const block_length,
      ::yampi::environment const& environment)
      : mpi_datatype_{
          derive(
            base_datatype, displacement_first, displacement_last, block_length,
            environment)}
    { }

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    [[deprecated]]
    datatype(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment)
      : mpi_datatype_{
          derive(
            datatype_first, datatype_last, byte_displacement_first, block_length_first,
            environment)}
    { }
    // *** end of DEPRECATED CONSTRUCTORS *** //

   private:
    template <typename DerivedDatatype>
    MPI_Datatype duplicate(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
      int const error_code
        = MPI_Type_dup(base_datatype.mpi_datatype(), std::addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::datatype::duplicate", environment);
    }

    // MPI_Type_contiguous
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype, ::yampi::count const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_contiguous_c(
            count.mpi_count(), base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_contiguous(
            count.int_mpi_count(), base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_vector
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::strided_block const& block, ::yampi::count const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_vector_c(
            count.mpi_count(), block.length().mpi_count(), block.stride().mpi_count(),
            base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_vector(
            count.int_mpi_count(), block.length().int_mpi_count(), block.stride().int_mpi_count(),
            base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hvector
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_strided_block const& block, ::yampi::count const count,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hvector_c(
            count.mpi_count(), block.length().mpi_count(), block.stride_bytes().mpi_count(),
            base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hvector(
            count.int_mpi_count(), block.length().int_mpi_count(), block.stride_bytes().mpi_byte_displacement(),
            base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_indexed
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::flexible_blocks const& blocks,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_indexed_c(
            blocks.count().mpi_count(),
            reinterpret_cast<MPI_Count const*>(blocks.length_begin()),
            reinterpret_cast<MPI_Count const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_indexed(
            blocks.count().int_mpi_count(), blocks.length_begin(), blocks.displacement_begin(),
            base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hindexed
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_flexible_blocks const& blocks,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hindexed_c(
            blocks.count().mpi_count(),
            reinterpret_cast<MPI_Count const*>(blocks.length_begin()),
            reinterpret_cast<MPI_Count const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hindexed(
            blocks.count().int_mpi_count(), blocks.length_begin(),
            reinterpret_cast<MPI_Aint const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_indexed_block
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::fixed_blocks const& blocks,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_indexed_block_c(
            blocks.count().mpi_count(), blocks.length().mpi_count(),
            reinterpret_cast<MPI_Count const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_indexed_block(
            blocks.count().int_mpi_count(), blocks.length().int_mpi_count(),
            blocks.displacement_begin(),
            base_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

# if MPI_VERSION >= 3
    // MPI_Type_create_hindexed_block
    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_fixed_blocks const& blocks,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
#   if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hindexed_block_c(
            blocks.count().mpi_count(), blocks.length().mpi_count(),
            reinterpret_cast<MPI_Count const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
#   else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_hindexed_block(
            blocks.count().int_mpi_count(), blocks.length().int_mpi_count(),
            reinterpret_cast<MPI_Aint const*>(blocks.displacement_begin()),
            base_datatype.mpi_datatype(), std::addressof(result));
#   endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }
# endif // MPI_VERSION >= 3

    // MPI_Type_create_struct
    template <typename Datatype>
    MPI_Datatype derive(
      ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& typed_blocks,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_struct_c(
            typed_blocks.count().mpi_count(),
            reinterpret_cast<MPI_Count const*>(typed_blocks.length_begin()),
            reinterpret_cast<MPI_Count const*>(typed_blocks.displacement_begin()),
            reinterpret_cast<MPI_Datatype const*>(typed_blocks.datatype_begin()),
            std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_struct(
            typed_blocks.count().int_mpi_count(), typed_blocks.length_begin(),
            reinterpret_cast<MPI_Aint const*>(typed_blocks.displacement_begin()),
            reinterpret_cast<MPI_Datatype const*>(typed_blocks.datatype_begin()),
            std::addressof(result));
# endif // MPI_VERSION >= 4

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
# if MPI_VERSION >= 4
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           ::yampi::count const>::value),
        "The value type of ContiguousIterator1 must be convertible to ::yampi::count const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           ::yampi::count const>::value),
        "The value type of ContiguousIterator2 must be convertible to ::yampi::count const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator3>::value_type,
           ::yampi::count const>::value),
        "The value type of ContiguousIterator3 must be convertible to ::yampi::count const");
# else // MPI_VERSION >= 4
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           int const>::value),
        "The value type of ContiguousIterator1 must be convertible to int const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator3>::value_type,
           int const>::value),
        "The value type of ContiguousIterator3 must be convertible to int const");
# endif // MPI_VERSION >= 4

      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_subarray_c(
            array_size_last - array_size_first,
            reinterpret_cast<MPI_Count const*>(std::addressof(*array_size_first)),
            reinterpret_cast<MPI_Count const*>(std::addressof(*array_subsize_first)),
            reinterpret_cast<MPI_Count const*>(std::addressof(*array_start_index_first)),
            MPI_ORDER_C, array_element_datatype.mpi_datatype(), std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_subarray(
            array_size_last - array_size_first,
            std::addressof(*array_size_first),
            std::addressof(*array_subsize_first),
            std::addressof(*array_start_index_first),
            MPI_ORDER_C, array_element_datatype.mpi_datatype(), std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    template <typename DerivedDatatype>
    MPI_Datatype derive(
      ::yampi::datatype_base<DerivedDatatype> const& old_datatype,
      ::yampi::bounds const& new_bounds,
      ::yampi::environment const& environment)
    {
      MPI_Datatype result;
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_resized_c(
            old_datatype.mpi_datatype(),
            new_bounds.lower_bound().mpi_extent(),
            new_bounds.extent().mpi_extent(),
            std::addressof(result));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Type_create_resized(
            old_datatype.mpi_datatype(),
            new_bounds.lower_bound().mpi_aint_mpi_extent(),
            new_bounds.extent().mpi_aint_mpi_extent(),
            std::addressof(result));
# endif // MPI_VERSION >= 4

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // *** DEPRECATED derive FUNCTIONS *** //
    // MPI_Type_indexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    [[deprecated]]
    typename std::enable_if<
      not std::is_same<
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
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           int const>::value),
        "The value type of ContiguousIterator1 must be convertible to int const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_indexed(
            displacement_last - displacement_first,
            std::addressof(*block_length_first),
            std::addressof(*displacement_first),
            base_datatype.mpi_datatype(), std::addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    [[deprecated]]
    typename std::enable_if<
      std::is_same<
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
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           int const>::value),
        "The value type of ContiguousIterator2 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_hindexed(
            byte_displacement_last - byte_displacement_first,
            std::addressof(*block_length_first),
            reinterpret_cast<MPI_Aint const*>(
              std::addressof(*byte_displacement_first)),
            base_datatype.mpi_datatype(), std::addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

    // MPI_Type_create_indexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    [[deprecated]]
    typename std::enable_if<
      not std::is_same<
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
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           int const>::value),
        "The value type of ContiguousIterator must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_indexed_block(
            displacement_last - displacement_first,
            block_length, std::addressof(*displacement_first),
            base_datatype.mpi_datatype(), std::addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }

# if MPI_VERSION >= 3
    // MPI_Type_create_hindexed_block
    template <typename DerivedDatatype, typename ContiguousIterator>
    [[deprecated]]
    typename std::enable_if<
      std::is_same<
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
              std::addressof(*byte_displacement_first)),
            base_datatype.mpi_datatype(), std::addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }
# endif // MPI_VERSION >= 3

    // MPI_Type_create_struct
    template <
      typename ContiguousIterator1, typename ContiguousIterator2,
      typename ContiguousIterator3>
    [[deprecated]]
    MPI_Datatype derive(
      ContiguousIterator1 const datatype_first,
      ContiguousIterator1 const datatype_last,
      ContiguousIterator2 const byte_displacement_first,
      ContiguousIterator3 const block_length_first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator1>::value_type,
           ::yampi::datatype const>::value),
        "The value type of ContiguousIterator1 must be convertible to"
        " yampi::datatype const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator2>::value_type,
           ::yampi::byte_displacement const>::value),
        "The value type of ContiguousIterator2 must be convertible to"
        " yampi::byte_displacement const");
      static_assert(
        (std::is_convertible<
           typename std::iterator_traits<ContiguousIterator3>::value_type,
           int const>::value),
        "The value type of ContiguousIterator3 must be convertible to int const");

      MPI_Datatype result;
      int const error_code
        = MPI_Type_create_struct(
            datatype_last - datatype_first,
            std::addressof(*block_length_first),
            reinterpret_cast<MPI_Aint const*>(
              std::addressof(*byte_displacement_first)),
            reinterpret_cast<MPI_Datatype const*>(std::addressof(*datatype_first)),
            std::addressof(result));

      return error_code == MPI_SUCCESS
        ? commit(result, environment)
        : throw ::yampi::error(error_code, "yampi::datatype::derive", environment);
    }
    // *** end of DEPRECATED derive FUNCTIONS *** //

    MPI_Datatype commit(
      MPI_Datatype const& mpi_datatype,
      ::yampi::environment const& environment) const
    {
      MPI_Datatype result = mpi_datatype;
      int const error_code = MPI_Type_commit(std::addressof(result));

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::datatype::commit", environment);
    }

   public:
    bool operator==(datatype const& other) const noexcept
    { return mpi_datatype_ == other.mpi_datatype_; }

   private:
    friend base_type;

    bool do_is_null() const noexcept
    { return mpi_datatype_ == MPI_DATATYPE_NULL; }

    MPI_Datatype do_mpi_datatype() const noexcept
    { return mpi_datatype_; }

   public:
    void free(::yampi::environment const& environment)
    {
      if (mpi_datatype_ == MPI_DATATYPE_NULL
          or ::yampi::datatype_detail::is_predefined_mpi_datatype(mpi_datatype_))
        return;

      int const error_code = MPI_Type_free(std::addressof(mpi_datatype_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::datatype::free", environment);
    }

    void reset(MPI_Datatype const& mpi_datatype, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = mpi_datatype;
    }

    void reset(datatype&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = std::move(other.mpi_datatype_);
      other.mpi_datatype_ = MPI_DATATYPE_NULL;
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

    // MPI_Type_indexed
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::flexible_blocks const& blocks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, blocks, environment);
    }

    // MPI_Type_create_hindexed
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_flexible_blocks const& blocks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, blocks, environment);
    }

    // MPI_Type_create_indexed_block
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::fixed_blocks const& blocks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, blocks, environment);
    }

# if MPI_VERSION >= 3
    // MPI_Type_create_hindexed_block
    template <typename DerivedDatatype>
    void reset(
      ::yampi::datatype_base<DerivedDatatype> const& base_datatype,
      ::yampi::heterogeneous_fixed_blocks const& blocks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(base_datatype, blocks, environment);
    }
# endif // MPI_VERSION >= 3

    // MPI_Type_create_struct
    template <typename Datatype>
    void reset(
      ::yampi::heterogeneous_typed_flexible_blocks<Datatype> const& blocks,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(blocks, environment);
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
      ::yampi::bounds const& new_bounds,
      ::yampi::environment const& environment)
    {
      free(environment);
      mpi_datatype_ = derive(old_datatype, new_bounds, environment);
    }

    ::yampi::count size(::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 4
      MPI_Count result;
      int const error_code = MPI_Type_size_c(mpi_datatype_, std::addressof(result));
# elif MPI_VERSION >= 3
      MPI_Count result;
      int const error_code = MPI_Type_size_x(mpi_datatype_, std::addressof(result));
# else
      int result;
      int const error_code = MPI_Type_size(mpi_datatype_, std::addressof(result));
# endif
      return error_code == MPI_SUCCESS
        ? ::yampi::count(result)
        : throw ::yampi::error(
            error_code, "yampi::datatype::size", environment);
    }

    ::yampi::bounds bounds(::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 4
      MPI_Count lower_bound, extent;
      int const error_code
        = MPI_Type_get_extent_c(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# elif MPI_VERSION >= 3
      MPI_Count lower_bound, extent;
      int const error_code
        = MPI_Type_get_extent_x(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# else
      MPI_Aint lower_bound, extent;
      int const error_code
        = MPI_Type_get_extent(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# endif
      return error_code == MPI_SUCCESS
        ? ::yampi::bounds(::yampi::extent(lower_bound), ::yampi::extent(extent))
        : throw ::yampi::error(
            error_code, "yampi::datatype::bounds", environment);
    }

    ::yampi::bounds true_bounds(::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 4
      MPI_Count lower_bound, extent;
      int const error_code
        = MPI_Type_get_true_extent_c(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# elif MPI_VERSION >= 3
      MPI_Count lower_bound, extent;
      int const error_code
        = MPI_Type_get_true_extent_x(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# else
      MPI_Aint lower_bound, extent;
      int const error_code
        = MPI_Type_get_true_extent(
            mpi_datatype_, std::addressof(lower_bound), std::addressof(extent));
# endif
      return error_code == MPI_SUCCESS
        ? ::yampi::bounds(::yampi::extent(lower_bound), ::yampi::extent(extent))
        : throw ::yampi::error(
            error_code, "yampi::datatype::true_bounds", environment);
    }

    // *** DEPRECATED reset FUNCTIONS *** //
    // MPI_Type_indexed, MPI_Type_create_hindexed
    template <typename DerivedDatatype, typename ContiguousIterator1, typename ContiguousIterator2>
    [[deprecated]]
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
    [[deprecated]]
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
    [[deprecated]]
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
    // *** end of DEPRECATED reset FUNCTIONS *** //

    void swap(datatype& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Datatype>::value)
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

