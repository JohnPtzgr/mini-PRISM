
-- function to_bits(num)
--   if num == 0 then
--     return tostring(num):reverse() .. ':0'
--   end
--   bits = tostring(num):reverse() .. ':'
--   while num > 0 do
--     bits = bits .. tostring(num & 1)
--     num = num // 2
--   end
--   return bits:reverse()
-- end

-- function table_as_str(t)
--   srepr = ''
--   for k, v in pairs(t) do
--     srepr = srepr .. tostring(v)
--   end
--   return srepr
-- end

function norm8BitU(x)
  return math.min(255, math.max(0, x))
end


function tableEq(t1, t2)
  for k, v in pairs(t1) do
    if v ~= t2[k] then
      return false
    end
  end
  return #t1 == #t2
end

---comment
---@param str string
function as_bits(str, stream, flag)
  stream = {}
  flag = 0
  for i=1,#str do
    chr = str:byte(i) - 48
    for j=1,6 do
      if flag == 0 and (chr & 32) == 32 then
        flag = j + 6 * i - 6
      else
        stream[6 * i - 6 + j - flag] = (chr & 32) // 32
      end
      chr = chr * 2
    end
  end
  return stream
end


function LoadImg(enc_img_s, enc_tree_s, enc_cols_s)
  tree = {}
  cols = {}
  cols_bits = as_bits(enc_cols_s)
  for i=0,(#cols_bits // 24)-1 do
    cols[i] = {0, 0, 0}
    function setcols(c)
      for j=1,8 do
        cols[i][c] = cols[i][c] + (2 ^ (8 - j)) * cols_bits[24 * i + 8 * c - 8 + j]
      end
    end
    setcols(1)
    setcols(2)
    setcols(3)
  end
  tcol = cols[0]
  tree_bits = as_bits(enc_tree_s)
  ets_key = 1
  cnt = 0
  idx = 1
  for i, b in ipairs(tree_bits) do
    if b ~= 0 then
      idx = i
      break
    end
  end

  for i=idx, #tree_bits do
    tb = tree_bits[i]
    if tb == 0 then
      cnt = cnt + 1
      col = cols[cnt]
      if col == nil then
        break
      end
      tree[ets_key] = cols[cnt]
      ets_key = ets_key + 1
    else
      ets_key = ets_key * 2
    end
  end

  px_bits = as_bits(enc_img_s)

  idx = 1

  for i, tb in ipairs(px_bits) do
    if tb ~= 0 then
      idx = i
      break
    end
  end

  px_cols = {}
  es_key = 0
  cnt = 0

  while idx <= #px_bits do
    tb = px_bits[idx]
    idx = idx + 1
    es_key = es_key * 2 + tb
    if tree[es_key] ~= nil then
      cnt = cnt + 1
      px_cols[cnt] = tree[es_key]
      es_key = 1
    end
  end
  state = {}
  for i=0,256 do
    state[i] = 0
  end
  return {tree, px_cols, tcol, state, {}, #px_cols + 1}
end

function drawImg(img, x, y, w, h, cond)
  tree = img[1]
  px_cols = img[2]
  brush_token = img[3]
  state = img[4]
  buttons = img[5]
  user_section = img[6]

  x0 = 0
  y0 = 0
  bw = w
  bh = h
  token_b = 0

  str_len = 0
  cond_len = 0
  skp = 0

  function dtext(c, s)
    if str_len == 0 or skp == 1 then
      skp = 0
      return
    end
    if col[c] == 0 then
      s = string.format('%.0f', state[col[c + 1]])
      skp = 1
    else
      s = string.char(col[c])
    end
    str_len = str_len - 1
    screen.drawText(x + x0, y + y0, s)
    x0 = x0 + 5 * #s
  end

  dcolr = 0
  dcolg = 0
  dcolb = 0
  dalpha = 255
  proportional = true
  kX = w
  kY = h

  idx = 1


  while idx <= #px_cols do
    -- -- print(state[3], state[4], state[5], state[24])
    -- if idx >= 112 then
    --   print("READING USERDATA")
    -- end
    col = px_cols[idx]
    idx = idx + 1
    col1 = col[1]
    col2 = col[2]
    col3 = col[3]
    stcol2 = state[col2]
    stcol3 = state[col3]
    if not tableEq(col, brush_token) or token_b == 1 or str_len ~= 0 then
      if cond_len == 0 then
        if token_b == 0 then
          if str_len == 0 then
            screen.setColor(norm8BitU(col1 + dcolr),
                            norm8BitU(col2 + dcolg),
                            norm8BitU(col3 + dcolb),
                            norm8BitU(dalpha))
            if bw + bh > 0 then
              screen.drawRectF(x + x0, y + y0, bw, bh)
            else
              -- print("Draw COL")
            end
            x0 = x0 + bw
          else
            dtext(1)
            dtext(2)
            dtext(3)
          end
        else
          if col1 & 15 == 0 then
            x0 = (col1 // 16 + 1) * col2 * w
            y0 = (col1 // 16 + 1) * col3 * h
          elseif col1 & 15 == 1 then
            x0 = x0 + (col1 // 16 + 1) * (col2 - 128) * w
            y0 = y0 + (col1 // 16 + 1) * (col3 - 128) * h
          elseif col1 & 15 == 2 then
            bw = (col1 // 16 + 1) * col2 * w
            bh = (col1 // 16 + 1) * col3 * h
            proportional = true
            kX, kY = w, h
          elseif col1 & 15 == 3 then
            x0 = x0 + (col1 // 16 + 1) * (col2 - 128)
            y0 = y0 + (col1 // 16 + 1) * (col3 - 128)
          elseif col1 & 15 == 4 then
            bw = (col1 // 16 + 1) * col2
            bh = (col1 // 16 + 1) * col3
            proportional = false
            kX, kY = 1, 1
          elseif col1 & 15 == 5 then
            if col1 // 16 == 0 then
              screen.drawRect(x + x0,
                              y + y0,
                              bw + col2 * kX,
                              bh + col3 * kY)
            elseif col1 // 16 == 1 then
              screen.drawRectF(x + x0,
                               y + y0,
                               bw + col2 * kX,
                               bh + col3 * kY)
            elseif col1 // 16 == 2 then
              screen.drawLine(x + x0,
                              y + y0,
                              x + x0 + bw + col2 * kX,
                              y + y0 + bh + col3 * kY)
            elseif col1 // 16 == 3 then
              screen.drawCircle(x + x0,
                                y + y0,
                                bw + col2 * kX)
            elseif col1 // 16 == 4 then
              screen.drawCircleF(x + x0,
                                 y + y0,
                                 bw + col2 * kX)
            elseif col1 // 16 == 5 then
              screen.drawTriangle(x + x0,
                                  y + y0,
                                  x + x0 + bw,
                                  y + y0 + bh,
                                  x + x0 + col2 * kX,
                                  y + y0 + col3 * kY)
            elseif col1 // 16 == 6 then
              screen.drawTriangleF(x + x0,
                                   y + y0,
                                   x + x0 + bw,
                                   y + y0 + bh,
                                   x + x0 + col2 * kX,
                                   y + y0 + col3 * kY)
            end
          elseif col1 & 15 == 6 then
            str_len = col1 // 16 + 1
            dtext(2)
            dtext(3)
          elseif col1 & 15 == 7 then
            dalpha = dalpha + (col3 & 127) - 64
            acolx = math.pi * (1 + col1 // 16 * 2) / 16
            acoly = math.pi * (1 + col2 // 32 * 2) / 16
            colrad = (col2 & 31) * 2 + col3 // 128
            dcolr = dcolr + colrad * math.cos(acolx) * math.sin(acoly)
            dcolg = dcolg + colrad * math.sin(acolx) * math.sin(acoly)
            dcolb = dcolb + colrad * math.cos(acoly)
          elseif col1 & 15 == 8 then
            buttons[col1 // 16] = {x0, y0, bw, bh, col2, col3}
          elseif col1 & 15 == 9 then
            flag = state[16 + col1 // 16]
            if col2 // 64 == 0 then
              cond = flag & col3 > 0
            elseif col2 // 64 == 1 then
              cond = flag & col3 > 0
            elseif col2 // 64 == 2 then
              cond = flag & col3 == 0
            else
              cond = flag == col3
            end
            if cond then
              cond_len = col2 & 63
            end
          elseif col1 & 15 == 10 then
            if col1 // 16 == 0 then
              state[col2] = col3
            elseif col1 // 16 == 1 then
              state[col2] = stcol2 + stcol3
            elseif col1 // 16 == 2 then
              state[col2] = stcol2 - stcol3
            elseif col1 // 16 == 3 then
              state[col2] = stcol2 * stcol3
            elseif col1 // 16 == 4 then
              state[col2] = stcol2 // stcol3
            elseif col1 // 16 == 5 then
              state[col2] = stcol2 | stcol3
            elseif col1 // 16 == 6 then
              state[col2] = stcol2 & stcol3
            elseif col1 // 16 == 7 then
              state[col2] = stcol2 ^ stcol3
            elseif col1 // 16 == 8 then
              state[col2] = col3 - 128 - stcol2
            elseif col1 // 16 == 9 then
              state[col2] = col3 - 128 + stcol2
            elseif col1 // 16 == 10 then
              state[col2] = stcol3
            elseif col1 // 16 == 11 then
              state[col2] = stcol2 << (col3 - 128)
            elseif col1 // 16 == 12 then
              state[256] = (state[256] & 252) + ((stcol2 < col3) and 2 or 0) +
                           ((stcol2 > col3) and 1 or 0)
            elseif col1 // 16 == 13 then
              state[col2] = stcol2 % stcol3
            elseif col1 // 16 == 14 then
              idx = idx + ((state[256] & 3 == 0) and (stcol2 * 256 + stcol3 - 32768) or 0)
            elseif col1 // 16 == 15 then
              px_cols[user_section + stcol2 // 256] = {stcol2 & 255, stcol3 // 256, stcol3 & 255}
              -- print(px_cols[user_section + stcol2 // 256][1],
              --       px_cols[user_section + stcol2 // 256][2],
              --       px_cols[user_section + stcol2 // 256][3],
              --       brush_token[1],
              --       brush_token[2],
              --       brush_token[3],
              --       user_section,
              --       #px_cols,
              --       tableEq(px_cols[user_section + stcol2 // 256], brush_token))
              state[col2] = stcol2 + 256
            end
          elseif col1 & 15 == 11 then
            if col1 // 16 == 0 then
              bw = stcol2 * kX
              bh = stcol3 * kY
            elseif col1 // 16 == 1 then
              x0 = stcol2 * kX
              y0 = stcol3 * kY
            elseif col1 // 16 == 2 then
              x0 = x0 + stcol2 * kX
              y0 = y0 + stcol3 * kY
            end
          end
        end
      else
        cond_len = cond_len - 1
      end
      token_b = 0
    else
      token_b = 1
    end
  end
end

function checkButtons(img, x, y, kX, kY, tX, tY)
  for i, bt in pairs(img[5]) do
    rX, rY = tX - x - bt[1] - bt[3] / 2, tY - y - bt[2] - bt[4] / 2
    if math.abs(rX/kX) < bt[3] / 2 and math.abs(rY/kY) < bt[4] / 2 then
      img[4][16 + i] = img[4][16 + i] | bt[5]
    else
      img[4][16 + i] = img[4][16 + i] & ~bt[6]
    end
  end
end
