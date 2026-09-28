-- Keep the editorial choices in Markdown; emit deliberately simple LaTeX.
local function latex(blocks)
  return pandoc.write(pandoc.Pandoc(blocks), 'latex'):gsub('%s+$', '')
end
local function equation(math, numbered)
  local env = numbered and 'equation' or 'equation*'
  return pandoc.RawInline('latex', '\\begin{' .. env .. '}\n' .. math.text:match('^%s*(.-)%s*$') .. '\n\\end{' .. env .. '}')
end
local function has_class(el, name)
  return el.classes:includes(name)
end

return {
  { Div = function(div)
      if has_class(div, 'numbered') then
        return div:walk({Math = function(m)
          if m.mathtype == 'DisplayMath' then return equation(m, true) end
        end}).content
      elseif has_class(div, 'course-table') then
        return div:walk({Table = function(t)
          for k, v in pairs(div.attributes) do t.attributes[k] = v end
          return t
        end}).content
      end
    end },
  { Math = function(m)
      if m.mathtype == 'DisplayMath' then return equation(m, false) end
      return pandoc.RawInline('latex', '$' .. m.text .. '$')
    end,
    Str = function(s)
      s.text = s.text:gsub('—', '-'):gsub('–', '-')
      return s
    end,
    Image = function(im)
      local src = im.src:gsub('^%.%./img/', 'img/')
      local width = im.attributes.width or '\\linewidth'
      width = width:gsub('(%d+)%%', function(n) return tonumber(n)/100 .. '\\linewidth' end)
      return pandoc.RawInline('latex', '\\includegraphics[width=' .. width .. ']{' .. src .. '}')
    end },
  { Table = function(t)
      local cols = t.attributes.columns
      if not cols then
        local specs = {}
        for _ = 1, #t.colspecs do specs[#specs+1] = '>{\\raggedright\\arraybackslash}p{' .. (0.9/#t.colspecs) .. '\\linewidth}' end
        cols = '@{}' .. table.concat(specs) .. '@{}'
      end
      local result = {'\\begingroup', '\\setlength{\\tabcolsep}{' .. (t.attributes.tabcolsep or '3pt') .. '}'}
      if t.attributes['font-size'] == 'small' then result[#result+1] = '\\small' end
      result[#result+1] = '\\begin{tabular}{' .. cols .. '}\n\\toprule'
      local function rows(rs)
        for _, row in ipairs(rs) do
          local cells = {}
          for _, cell in ipairs(row.cells) do
            assert(cell.row_span == 1 and cell.col_span == 1, 'Merged table cells require an explicit renderer')
            cells[#cells+1] = latex(cell.contents)
          end
          result[#result+1] = table.concat(cells, ' & ') .. ' \\\\'
        end
      end
      rows(t.head.rows)
      if #t.head.rows > 0 then result[#result+1] = '\\midrule' end
      for _, body in ipairs(t.bodies) do rows(body.head); rows(body.body) end
      rows(t.foot.rows)
      result[#result+1] = '\\bottomrule\n\\end{tabular}\n\\endgroup'
      return pandoc.RawBlock('latex', table.concat(result, '\n'))
    end,
    Header = function(h)
      local commands = {'section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph', 'subparagraph'}
      local space = h.attributes.needspace or '4'
      local title = latex({pandoc.Plain(h.content)})
      return pandoc.RawBlock('latex', '\\Needspace{' .. space .. '\\baselineskip}\n\\' .. commands[h.level] .. '{' .. title .. '}')
    end,
    HorizontalRule = function()
      return pandoc.RawBlock('latex', '\\begin{center}\\rule{0.5\\linewidth}{0.5pt}\\end{center}')
    end },
  { Div = function(div)
      if has_class(div, 'center') then
        local content = {pandoc.RawBlock('latex', '\\begin{center}')}
        for _, b in ipairs(div.content) do content[#content+1] = b end
        content[#content+1] = pandoc.RawBlock('latex', '\\end{center}')
        return content
      end
    end }
}
