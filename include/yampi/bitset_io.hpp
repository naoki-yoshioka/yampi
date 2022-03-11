#ifndef YAMPI_BITSET_IO_HPP
# define YAMPI_BITSET_IO_HPP

# include <boost/config.hpp>

# include <istream>
# include <ostream>
# include <ios>
# include <string>
# include <locale>

# include <yampi/bitset.hpp>


namespace yampi
{
  template <typename Character, typename CharacterTraits, std::size_t N, typename DataElement>
  inline std::basic_ostream<Character, CharacterTraits>& operator<<(
    std::basic_ostream<Character, CharacterTraits>& output_stream, ::yampi::bitset<N, DataElement> const& bits)
  {
    std::ctype<Character> const& character_type = std::use_facet< std::ctype<Character> >(output_stream.getloc());
    return output_stream << bits.to_string(character_type.widen('0'), character_type.widen('1'));
  }

  template <typename Character, typename CharacterTraits, std::size_t N, typename DataElement>
  inline std::basic_istream<Character, CharacterTraits>& operator>>(
    std::basic_istream<Character, CharacterTraits>& input_stream, ::yampi::bitset<N, DataElement>& bits)
  {
    typedef typename CharacterTraits::char_type char_type;
    typedef typename CharacterTraits::int_type int_type;
    typedef std::basic_istream<Character, CharacterTraits> input_stream_type;
    typedef typename input_stream_type::ios_base ios_base_type;

    typedef std::basic_string<Character, CharacterTraits> string_type;
    typedef typename string_type::size_type size_type;

    string_type tmp;
    tmp.reserve(N);

    char_type const zero = input_stream.widen('0');
    char_type const one = input_stream.widen('1');

    typename ios_base_type::iostate io_state = ios_base_type::goodbit;
    typename input_stream_type::sentry sentry(input_stream);
    if (sentry)
    {
      try
      {
        for (size_type index = static_cast<size_type>(0u); index < N; ++index)
        {
          BOOST_STATIC_CONSTEXPR int_type eof = CharacterTraits::eof();

          int_type const int_character = input_stream.rdbuf()->sbumpc();
          if (CharacterTraits::eq_int_type(int_character, eof))
          {
            io_state |= ios_base_type::eofbit;
            break;
          }
          else
          {
            char_type const character = CharacterTraits::to_char_type(int_character);
            if (CharacterTraits::eq(character, zero))
              tmp.push_back(zero);
            else if (CharacterTraits::eq(character, one))
              tmp.push_back(one);
            else if (CharacterTraits::eq_int_type(input_stream.rdbuf()->sputbackc(character), eof))
            {
              io_state |= ios_base_type::failbit;
              break;
            }
          }
        }
      }
      catch(...)
      { input_stream.setstate(ios_base_type::badbit); }
    }

    if (tmp.empty() && (N != 0))
      io_state |= ios_base_type::failbit;
    else
      bitset.from_string(tmp, static_cast<size_type>(0u), string_type::npos, zero, one);

    if (io_state != ios_base_type::goodbit)
      input_stream.setstate(io_state);

    return input_stream;
  }
}


#endif

