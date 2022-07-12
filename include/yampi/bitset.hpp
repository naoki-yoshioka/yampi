#ifndef YAMPI_BITSET_HPP
# define YAMPI_BITSET_HPP

# include <cstddef>
# include <string>
# include <array>
# include <iterator>
# include <algorithm>
# include <functional>
# include <limits>
# include <type_traits>
# include <stdexcept>
# include <memory>

# include <yampi/datatype.hpp>
# include <yampi/predefined_datatype.hpp>
# include <yampi/environment.hpp>


namespace yampi
{
  template <std::size_t N, typename DataElement = unsigned long>
  class bitset
  {
    static_assert(std::is_integral<DataElement>::value and std::is_unsigned<DataElement>::value, "DataElement should be an unsigned integral type");

    constexpr std::size_t num_data_elements_
      = N / std::numeric_limits<DataElement>::digits
        + (N % std::numeric_limits<DataElement>::digits == 0u ? 0u : 1u);

    std::array<DataElement, num_data_elements_> data_;

    static constexpr std::size_t data_element_index(std::size_t const position) noexcept
    { return position / std::numeric_limits<DataElement>::digits; }

    static constexpr std::size_t bit_position(std::size_t const position) noexcept
    { return position % std::numeric_limits<DataElement>::digits; }

    static constexpr std::size_t single_bit_mask(std::size_t const position) noexcept
    { return static_cast<DataElement>(1u) << bitset::bit_position(position); }

    DataElement& data_element(std::size_t const position) noexcept
    { return data_[bitset::data_element_index(position)]; }

    constexpr DataElement data_element(std::size_t const position) const noexcept
    { return data_[bitset::data_element_index(position)]; }

    constexpr bool unsafe_test(std::size_t const position) const noexcept
    { return (data_element(position) bitand bitset::single_bit_mask(position)) != static_cast<DataElement>(0u); }

    void unsafe_set(std::size_t const position) noexcept
    { data_element(position) |= bitset::single_bit_mask(position); }

    void unsafe_reset(std::size_t const position) noexcept
    { data_element(position) &= compl bitset::single_bit_mask(position); }

    void unsafe_flip(std::size_t const position) noexcept
    { data_element(position) ^= bitset::single_bit_mask(position); }

   public:
    static ::yampi::datatype const& datatype(::yampi::environment const& environment)
    {
      static ::yampi::datatype result(
        ::yampi::predefined_datatype<DataElement>(), static_cast<int>(num_data_elements_), environment);
      return result;
    }

    class reference
    {
      friend class bitset;

      DataElement* data_element_ptr_;
      std::size_t bit_position_;

      reference();

     public:
      reference(bitset& bits, std::size_t const position) noexcept
        : data_element_ptr_(YAMPI_addressof(bits.data_element(position))), bit_position_(bitset::bit_position(position))
      { }

      ~reference() noexcept { }

      // b[i] = x;
      reference& operator=(bool const boolean) noexcept
      {
        if (boolean)
          *data_element_ptr_ |= bitset::single_bit_mask(bit_position_);
        else
          *data_element_ptr_ &= compl bitset::single_bit_mask(bit_position_);
        return *this;
      }

      // b[i] = b[j];
      reference& operator=(reference const& other) noexcept
      {
        if (((*(other.data_element_ptr_)) bitand bitset::single_bit_mask(other.bit_position_))
            != static_cast<DataElement>(0u))
          *data_element_ptr_ |= bitset::single_bit_mask(bit_position_);
        else
          *data_element_ptr_ &= compl bitset::single_bit_mask(bit_position_);
        return *this;
      }

      // return the inverse of the referenced bit
      bool operator~() const noexcept
      { return (*data_element_ptr_ bitand bitset::single_bit_mask(bit_position_)) == static_cast<DataElement>(0u); }

      // return the referenced bit
      operator bool() const noexcept
      { return (*data_element_ptr_ bitand bitset::single_bit_mask(bit_position_)) != static_cast<DataElement>(0u); }

      // b[i].flip()
      reference& flip() noexcept
      { *data_element_ptr_ ^= bitset::single_bit_mask(bit_position_); return *this; }
    };
    friend class reference;

    constexpr bitset() noexcept : data_() { }

    bitset(unsigned long long const value) noexcept : data_() { from_ullong(value); }

    template <
      typename UnsignedInteger,
      typename = std::enable_if<std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value> >
    bitset(UnsignedInteger const value) noexcept
      : data_()
    { from_unsigned_integer(value); }

    template <typename Character, typename CharacterTraits, typename Allocator>
    explicit bitset(
      std::basic_string<Character, CharacterTraits, Allocator> const& string,
      typename std::basic_string<Character, CharacterTraits, Allocator>::size_type position = 0,
      typename std::basic_string<Character, CharacterTraits, Allocator>::size_type length
        = std::basic_string<Character, CharacterTraits, Allocator>::npos,
      Character const zero = Character('0'), Character const one = Character('1'))
    { from_string(string, position, length, zero, one); }

    template <typename Character>
    explicit bitset(
      Character const* string,
      typename std::basic_string<Character>::size_type length = std::basic_string<Character>::npos,
      Character const zero = Character('0'), Character const one = Character('1'))
    { from_characters(string, length, zero, one); }

    bool operator==(bitset const& other) const noexcept
    { return std::equal(std::begin(data_), std::end(data_), other.data_); }

    constexpr bool operator[](std::size_t const position) const
    { assert(position < N); return unsafe_test(position); }

    reference operator[](std::size_t const position)
    { assert(position < N): return reference(*this, position); }

    bool test(std::size_t const position) const
    {
      if (position >= N)
        throw std::out_of_range("out of range error at ::yampi::bitset::test");

      return unsafe_test(position);
    }

    bool all() const noexcept
    {
      std::size_t const offset = N % std::numeric_limits<DataElement>::digits;
      if (offset == static_cast<std::size_t>(0u))
        return std::all_of(std::begin(data_), std::end(data_), [](DataElement const& value) { return value == std::numeric_limits<DataElement>::max(); });

      return std::all_of(std::begin(data_), std::prev(std::end(data_)), [](DataElement const& value) { return value == std::numeric_limits<DataElement>::max(); })
        and (data_.back() == ((static_cast<DataElement>(1u) << offset) - static_cast<DataElement>(1u)));
    }

    bool any() const noexcept
    { return not std::all_of(std::begin(data_), std::end(data_), [](DataElement const& value) { return value == std::numeric_limits<DataElement>::min(); }); }

    bool none() const noexcept
    { return std::all_of(std::begin(data_), std::end(data_), [](DataElement const& value) { return value == std::numeric_limits<DataElement>::min(); }); }

    std::size_t count() const noexcept
    {
      std::size_t const offset = N % std::numeric_limits<DataElement>::digits;
      if (offset == static_cast<std::size_t>(0u))
        return partial_count(std::begin(data_), std::end(data_));

      std::size_t result = partial_count(std::begin(data_), std::prev(std::end(data_)));

      DataElement mask = static_cast<DataElement>(1u);
      for (int bit = 0; bit < offset; ++bit)
      {
        if ((data_.back() bitand mask) != std::numeric_limits<DataElement>::min())
          ++result;
        mask <<= 1u;
      }

      return result;
    }

   private:
    template <typename Iterator>
    std::size_t partial_count(Iterator const first, Iterator const last) const noexcept
    {
      return std::accumulate(
        first, last, static_cast<std::size_t>(0u),
        [](std::size_t partial_sum, DataElement const& data_element)
        {
          DataElement mask = static_cast<DataElement>(1u);
          for (int bit = 0; bit < std::numeric_limits<DataElement>::digits; ++bit)
          {
            if ((data_element bitand mask) != std::numeric_limits<DataElement>::min())
              ++partial_sum;
            mask <<= 1u;
          }

          return partial_sum;
        });
    }

   public:
    constexpr std::size_t size() const noexcept { return N; }

    bitset& operator&=(bitset const& other) noexcept
    { std::transform(std::begin(data_), std::end(data_), std::begin(other.data_), std::begin(data_), std::bit_and<DataElement>()); return *this; }

    bitset& operator|=(bitset const& other) noexcept
    { std::transform(std::begin(data_), std::end(data_), std::begin(other.data_), std::begin(data_), std::bit_or<DataElement>()); return *this; }

    bitset& operator^=(bitset const& other) noexcept
    { std::transform(std::begin(data_), std::end(data_), std::begin(other.data_), std::begin(data_), std::bit_xor<DataElement>()); return *this; }

    bitset operator~() noexcept
    { bitset result = *this; result.flip(); return result; }

    bitset& operator<<=(std::size_t const shift) noexcept
    {
      if (shift == static_cast<std::size_t>(0u))
        return *this;

      std::size_t const data_element_shift = shift / std::numeric_limits<DataElement>::digits;
      std::size_t const offset = shift % std::numeric_limits<DataElement>::digits;

      if (offset == static_cast<std::size_t>(0u))
        std::copy(data_.rbegin() + data_element_shift, data_.rend(), data_.rbegin());
      else
      {
        std::size_t const sub_offset = std::numeric_limits<DataElement>::digits - offset;
        std::transform(
          data_.rbegin() + data_element_shift + 1u, data_.rend(),
          data_.rbegin() + data_element_shift,
          data_.rbegin(),
          [offset, sub_offset](DataElement const& lhs, DataElement const& rhs)
          { return (lhs >> sub_offset) bitor (rhs << offset); });
        data_[data_element_shift] = data_.front() << offset;
      }

      std::fill(std::begin(data_), std::begin(data_) + data_element_shift, static_cast<DataElement>(0u));
      return *this;
    }

    bitset& operator>>=(std::size_t const shift) noexcept
    {
      if (shift == static_cast<std::size_t>(0u))
        return *this;

      std::size_t const data_element_shift = shift / std::numeric_limits<DataElement>::digits;
      std::size_t const offset = shift % std::numeric_limits<DataElement>::digits;

      if (offset == static_cast<std::size_t>(0u))
        std::copy(std::begin(data_) + data_element_shift, std::end(data_), std::begin(data_));
      else
      {
        std::size_t const sub_offset = std::numeric_limits<DataElement>::digits - offset;
        std::transform(
          std::begin(data_) + data_element_shift + 1u, std::end(data_),
          std::begin(data_) + data_element_shift,
          std::begin(data_),
          [offset, sub_offset](DataElement const& lhs, DataElement const& rhs)
          { return (lhs << sub_offset) bitor (rhs >> offset); });
        data_[num_data_elements_ - 1u - data_element_shift] = data_.back() >> offset;
      }

      std::fill(data_.rbegin(), data_.rbegin() + data_element_shift, static_cast<DataElement>(0u));
      return *this;
    }

    bitset& set() noexcept
    {
      std::size_t const offset = N % std::numeric_limits<DataElement>::digits;
      if (offset == static_cast<std::size_t>(0u))
      {
        std::fill(std::begin(data_), std::end(data_), std::numeric_limits<DataElement>::max());
        return *this;
      }

      std::fill(std::begin(data_), std::prev(std::end(data_)), std::numeric_limits<DataElement>::max());
      data_.back() = (static_cast<DataElement>(1u) << offset) - static_cast<DataElement>(1u);
      return *this;
    }

    bitset& set(std::size_t const position, bool value = true)
    {
      if (position >= N)
        throw std::out_of_range("out of range error at ::yampi::bitset::set");

      if (value)
        unsafe_set(position);
      else
        unsafe_reset(position);

      return *this;
    }

    bitset& reset() noexcept
    {
      std::fill(std::begin(data_), std::end(data_), std::numeric_limits<DataElement>::min());
      return *this;
    }

    bitset& reset(std::size_t const position)
    {
      if (position >= N)
        throw std::out_of_range("out of range error at ::yampi::bitset::set");

      unsafe_reset(position);
      return *this;
    }

    bitset& flip() noexcept
    {
      // This is implemented by using std::bit_not in C++14
      std::transform(
        std::begin(data_), std::end(data_), std::begin(data_),
        [](DataElement const& data_element) { return compl data_element; });

      std::size_t const offset = N % std::numeric_limits<DataElement>::digits;
      if (offset != static_cast<std::size_t>(0u))
        data_.back() &= (static_cast<DataElement>(1u) << offset) - static_cast<DataElement>(1u);

      return *this;
    }

    bitset& flip(std::size_t const position)
    {
      if (position >= N)
        throw std::out_of_range("out of range error at ::yampi::bitset::set");

      unsafe_flip(position);
      return *this;
    }

    template <
      typename Character = char, typename CharacterTraits = std::char_traits<Character>,
      typename Allocator = std::allocator<Character> >
    std::basic_string<Character, CharacterTraits, Allocator> to_string(
      Character const zero = Character('0'), Character const one = Character('1')) const
    {
      std::basic_string<Character, CharacterTraits, Allocator> result(N, zero);
      for (std::size_t position_plus_one = N; position_plus_one > static_cast<std::size_t>(0u); --position_plus_one)
        if (unsafe_test(position_plus_one - static_cast<std::size_t>(1u)))
          CharacterTraits::assign(result[N - position_plus_one], one);
    }

    template <typename Character, typename CharacterTraits, typename Allocator>
    std::basic_string<Character, CharacterTraits, Allocator> to_string(
      Character const zero = Character('0'), Character const one = Character('1')) const
    {
      std::basic_string<Character, CharacterTraits, Allocator> result(N, zero);
      for (std::size_t position_plus_one = N; position_plus_one > static_cast<std::size_t>(0u); --position_plus_one)
        if (unsafe_test(position_plus_one - static_cast<std::size_t>(1u)))
          CharacterTraits::assign(result[N - position_plus_one], one);

      return result;
    }

    template <typename Character, typename CharacterTraits>
    std::basic_string<Character, CharacterTraits, std::allocator<Character> > to_string(
      Character const zero = Character('0'), Character const one = Character('1')) const
    { return to_string<Character, CharacterTraits, std::allocator<Character> >(); }

    template <typename Character>
    std::basic_string<Character, std::char_traits<Character>, std::allocator<Character> > to_string(
      Character const zero = Character('0'), Character const one = Character('1')) const
    { return to_string<Character, std::char_traits<Character>, std::allocator<Character> >(zero, one); }

    std::basic_string<char, std::char_traits<char>, std::allocator<char> > to_string(
      char const zero = '0', char const one = '1') const
    { return to_string<char, std::char_traits<char>, std::allocator<char> >(zero, one); }

    unsigned long to_ulong() const { return to_unsigned_integer<unsigned long>(); }
    unsigned long to_ullong() const { return to_unsigned_integer<unsigned long long>(); }

    template <typename UnsignedInteger>
    typename std::enable_if<
      std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value,
      UnsignedInteger>::type
    to_unsigned_integer() const
    { return do_to_unsigned_integer<UnsignedInteger>::call(); }

   private:
    template <
      typename UnsignedInteger,
      int = (sizeof(UnsignedInteger) > sizeof(DataElement)
             ? +1
             : sizeof(UnsignedInteger) < sizeof(DataElement)
               ? -1
               : 0)>
    struct do_to_unsigned_integer;

    // sizeof(UnsignedInteger) > sizeof(DataElement)
    template <typename UnsignedInteger>
    struct do_to_unsigned_integer<UnsignedInteger, +1>
    {
      static UnsignedInteger call()
      {
        return do_to_unsigned_integer2<
          UnsignedInteger,
          sizeof(UnsignedInteger) / sizeof(DataElement),
          sizeof(UnsignedInteger) % sizeof(DataElement)>::call();
      }
    };

    // sizeof(UnsignedInteger) < sizeof(DataElement)
    template <typename UnsignedInteger>
    struct do_to_unsigned_integer<UnsignedInteger, -1>
    {
      static UnsignedInteger call()
      {
        // num_data_elements_ >= 1u
        if (std::any_of(std::next(std::begin(data_)), std::end(data_), [](DataElement const& value) { return value == static_cast<DataElement>(0u); }))
          throw std::overflow_error("bitset<N>::do_to_unsigned_integer<UnsignedInteger, -1>::call");

        constexpr DataElement mask
          = compl static_cast<DataElement>(std::numeric_limits<UnsignedInteger>::max());
        if ((data_.front() bitand mask) != static_cast<DataElement>(0u))
          throw std::overflow_error("bitset<N>::do_to_unsigned_integer<UnsignedInteger, -1>::call");

        return static_cast<UnsignedInteger>(data_.front());
      }
    };

    // sizeof(UnsignedInteger) == sizeof(DataElement)
    template <typename UnsignedInteger>
    struct do_to_unsigned_integer<UnsignedInteger, 0>
    {
      static UnsignedInteger call()
      {
        // num_data_elements_ >= 1u
        if (std::any_of(std::next(std::begin(data_)), std::end(data_), [](DataElement const& value) { return value == static_cast<DataElement>(0u); }))
          throw std::overflow_error("bitset<N>::do_to_unsigned_integer<UnsignedInteger, 0>::call");

        return static_cast<UnsignedInteger>(data_.front());
      }
    };

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) != 0u
    template <typename UnsignedInteger, std::size_t num_full_data_elements, std::size_t num_residual_bytes>
    struct do_to_unsigned_integer2
    {
      static UnsignedInteger call()
      { return do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes>::call(); }
    };

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) == 0u
    template <typename UnsignedInteger, std::size_t num_full_data_elements>
    struct do_to_unsigned_integer2<UnsignedInteger, num_full_data_elements, 0u>
    {
      static UnsignedInteger call()
      { return do_to_unsigned_integer2_2<UnsignedInteger, num_full_data_elements>::call(); }
    };

    template <
      typename UnsignedInteger, std::size_t num_full_data_elements, std::size_t num_residual_bytes,
      int = (num_data_elements_ > num_full_data_elements + 1u
             ? +1
             : num_data_elements_ == num_full_data_elements + 1u
               ? 0
               : -1)>
    struct do_to_unsigned_integer2_1;

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) != 0u,
    // num_data_elements_ > num_full_data_elements + 1u
    template <typename UnsignedInteger, std::size_t num_full_data_elements, std::size_t num_residual_bytes>
    struct do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, +1>
    {
      static UnsignedInteger call()
      {
        constexpr std::size_t num_residual_bits
          = num_residual_bytes * std::numeric_limits<unsigned char>::digits;

        if (std::any_of(
              std::begin(data_) + num_full_data_elements + 1u, std::end(data_), 
              [](DataElement const& value) { return value == static_cast<DataElement>(0u); }))
          throw std::overflow_error(
            "bitset<N>::do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, +1>::call");

        constexpr DataElement mask
          = compl ((static_cast<DataElement>(1u) << num_residual_bits)
                   - static_cast<DataElement>(1u));
        if ((data_[num_full_data_elements] bitand mask) != static_cast<DataElement>(0u))
          throw std::overflow_error(
            "bitset<N>::do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, +1>::call");

        return to_unsigned_integer_impl<UnsignedInteger, num_full_data_elements>::call(
          static_cast<UnsignedInteger>(data_[num_full_data_elements]));
      }
    };

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) != 0u,
    // num_data_elements_ == num_full_data_elements + 1u
    template <typename UnsignedInteger, std::size_t num_full_data_elements, std::size_t num_residual_bytes>
    struct do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, 0>
    {
      static UnsignedInteger call()
      {
        constexpr std::size_t num_residual_bits
          = num_residual_bytes * std::numeric_limits<unsigned char>::digits;

        constexpr DataElement mask
          = compl ((static_cast<DataElement>(1u) << num_residual_bits)
                   - static_cast<DataElement>(1u));
        if ((data_[num_full_data_elements] bitand mask) != static_cast<DataElement>(0u))
          throw std::overflow_error(
            "bitset<N>::do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, +1>::call");

        return to_unsigned_integer_impl<UnsignedInteger, num_full_data_elements>::call(
          static_cast<UnsignedInteger>(data_[num_full_data_elements]));
      }
    };

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) != 0u,
    // num_data_elements_ <= num_full_data_elements
    template <typename UnsignedInteger, std::size_t num_full_data_elements, std::size_t num_residual_bytes>
    struct do_to_unsigned_integer2_1<UnsignedInteger, num_full_data_elements, num_residual_bytes, -1>
    {
      static UnsignedInteger call()
      {
        return to_unsigned_integer_impl<UnsignedInteger, num_data_elements_>::call(
          static_cast<UnsignedInteger>(0u));
      }
    };

    template <
      typename UnsignedInteger, std::size_t num_full_data_elements,
      bool = (num_data_elements_ > num_full_data_elements)>
    struct do_to_unsigned_integer2_2;

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) == 0u,
    // num_data_elements_ > num_full_data_elements
    template <typename UnsignedInteger, std::size_t num_full_data_elements>
    struct do_to_unsigned_integer2_2<UnsignedInteger, num_full_data_elements, true>
    {
      static UnsignedInteger call()
      {
        if (std::any_of(
              std::begin(data_) + num_full_data_elements, std::end(data_),
              [](DataElement const& value) { return value == static_cast<DataElement>(0u); }))
          throw std::overflow_error(
            "bitset<N>::do_to_unsigned_integer2_2<UnsignedInteger, num_full_data_elements, checks_overflow>::call");

        return to_unsigned_integer_impl<UnsignedInteger, num_full_data_elements>::call(
          static_cast<UnsignedInteger>(0u));
      }
    };

    // sizeof(UnsignedInteger) > sizeof(DataElement), sizeof(UnsignedInteger) % sizeof(DataElement) == 0u,
    // num_data_elements_ <= num_full_data_elements
    template <typename UnsignedInteger, std::size_t num_full_data_elements>
    struct do_to_unsigned_integer2_2<UnsignedInteger, num_full_data_elements, false>
    {
      static UnsignedInteger call()
      {
        return to_unsigned_integer_impl<UnsignedInteger, num_data_elements_>::call(
          static_cast<UnsignedInteger>(0u));
      }
    };

    template <typename UnsignedInteger, std::size_t last_index>
    struct to_unsigned_integer_impl
    {
      static UnsignedInteger call(UnsignedInteger const initial_value)
      {
        return std::accumulate(
          data_.rbegin() + (num_data_elements_ - last_index), data_.rend(),
          initial_value,
          [](UnsignedInteger const partial_sum, DataElement const data_element)
          {
            return
              (partial_sum << std::numeric_limits<DataElement>::digits)
              + static_cast<UnsignedInteger>(data_element);
          });
      }
    };

   public:
    template <typename Character, typename CharacterTraits, typename Allocator>
    void from_string(
      std::basic_string<Character, CharacterTraits, Allocator> const& string,
      typename std::basic_string<Character, CharacterTraits, Allocator>::size_type position = 0,
      typename std::basic_string<Character, CharacterTraits, Allocator>::size_type length
        = std::basic_string<Character, CharacterTraits, Allocator>::npos,
      Character const zero = Character('0'), Character const one = Character('1'))
    {
      if (position > string.size())
        throw std::out_of_range("bitset::fron_string: position > string.size()");

      reset();

      typedef typename std::basic_string<Character, CharacterTraits, Allocator>::size_type size_type;
      size_type const effective_length = std::min(N, std::min(length, string.size() - position));

      for (size_type i = effective_length; i > static_cast<size_type>(0u); --i)
      {
        Character const character = string[position + effective_length - i];
        if (CharacterTraits::eq(character, one))
          unsafe_set(i - static_cast<size_type>(1u));
        else if (not CharacterTraits::eq(character, zero))
          throw std::invalid_argument("bitset::from_string: no zero and one character found");
      }
    }

    template <typename Character>
    void from_characters(
      Character const* string,
      typename std::basic_string<Character>::size_type length = std::basic_string<Character>::npos,
      Character const zero = Character('0'), Character const one = Character('1'))
    {
      from_string(
        length == std::basic_string<Character>::npos
          ? std::basic_string<Character>(string)
          : std::basic_string<Character>(string, length),
        0, length, zero, one);
    }

    void from_chars(
      char const* first, char const* last,
      char const zero = Character('0'), char const one = Character('1'))
    {
      typedef typename std::basic_string<char>::size_type size_type;
      size_type const effective_length = std::min(N, last - first);

      for (size_type i = effective_length; i > static_cast<size_type>(0); --i)
      {
        char const character = first[effective_length - i];
        if (character == one)
          unsafe_set(i - static_cast<size_type>(1u));
        else if (character == zero)
          unsafe_reset(i - static_cast<size_type>(1u));
      }
    }

    void from_ulong(unsigned long const value) noexcept { from_unsigned_integer(value); }
    void from_ullong(unsigned long long const value) noexcept { from_unsigned_integer(value); }

    template <typename UnsignedInteger>
    typename std::enable_if<
      std::is_integral<UnsignedInteger>::value and std::is_unsigned<UnsignedInteger>::value,
      void>::type
    from_unsigned_integer(UnsignedInteger const value) noexcept
    { do_from_unsigned_integer<UnsignedInteger>::call(value); }

   private:
    template <typename UnsignedInteger, bool = sizeof(UnsignedInteger) > sizeof(DataElement)>
    struct do_from_unsigned_integer;

    // sizeof(UnsignedInteger) <= sizeof(DataElement)
    template <typename UnsignedInteger>
    struct do_from_unsigned_integer<UnsignedInteger, false>
    { static void call(UnsignedInteger const value) { data_.front() = static_cast<DataElement>(value); } };

    // sizeof(UnsignedInteger) > sizeof(DataElement)
    template <typename UnsignedInteger>
    struct do_from_unsigned_integer<UnsignedInteger, true>
    { static void call(UnsignedInteger const value) { do_from_unsigned_integer2<0u>::call(value); } };

    template <std::size_t index>
    struct do_from_unsigned_integer2
    {
      template <typename UnsignedInteger>
      static void call(UnsignedInteger value)
      {
        data_[index] = static_cast<DataElement>(value bitand static_cast<UnsignedInteger>(std::numeric_limits<DataElement>::max()));

        value >>= std::numeric_limits<DataElement>::digits;
        if (value != static_cast<UnsignedInteger>(0u))
          do_from_unsigned_integer2<index + 1u>::call(value);
      }
    };

    template <>
    struct do_from_unsigned_integer2<num_data_elements_>
    { template <typename UnsignedInteger> static void call(UnsignedInteger const) { } };
  };

  template <std::size_t N, typename DataElement>
  inline bool operator!=(::yampi::bitset<N, DataElement> const& lhs, ::yampi::bitset<N, DataElement> const& rhs) noexcept
  { return not (lhs == rhs); }

  template <std::size_t N, typename DataElement>
  inline ::yampi::bitset<N, DataElement> operator<<(::yampi::bitset<N, DataElement> bits, std::size_t const shift) noexcept
  { return bits <<= shift; }

  template <std::size_t N, typename DataElement>
  inline ::yampi::bitset<N, DataElement> operator>>(::yampi::bitset<N, DataElement> bits, std::size_t const shift) noexcept
  { return bits >>= shift; }

  template <std::size_t N, typename DataElement>
  inline ::yampi::bitset<N, DataElement> operator&(::yampi::bitset<N, DataElement> lhs, ::yampi::bitset<N, DataElement> const& rhs) noexcept
  { return lhs &= rhs; }

  template <std::size_t N, typename DataElement>
  inline ::yampi::bitset<N, DataElement> operator|(::yampi::bitset<N, DataElement> lhs, ::yampi::bitset<N, DataElement> const& rhs) noexcept
  { return lhs |= rhs; }

  template <std::size_t N, typename DataElement>
  inline ::yampi::bitset<N, DataElement> operator^(::yampi::bitset<N, DataElement> lhs, ::yampi::bitset<N, DataElement> const& rhs) noexcept
  { return lhs ^= rhs; }
}


#endif

