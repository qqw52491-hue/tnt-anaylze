//! 语音风速解析:把语音识别文本转成风速值。
//! 支持 "正12" "正十二" "负123.2" "付5.8" "富三點二" "逆风三点五" 等。
//! 规则:取文本里最后一个数字(说了多个就覆盖前一个);
//! 该数字前面最近的符号词定正负,没有符号词默认正(顺风)。

fn cn_digit(c: char) -> Option<i64> {
    Some(match c {
        '0'..='9' => c as i64 - '0' as i64,
        '零' | '〇' => 0,
        '一' | '壹' => 1,
        '二' | '两' | '兩' | '贰' | '貳' => 2,
        '三' | '叁' | '參' => 3,
        '四' | '肆' => 4,
        '五' | '伍' => 5,
        '六' | '陆' | '陸' => 6,
        '七' | '柒' => 7,
        '八' | '捌' => 8,
        '九' | '玖' => 9,
        _ => return None,
    })
}

fn mult_val(c: char) -> Option<i64> {
    Some(match c {
        '十' | '拾' => 10,
        '百' | '佰' => 100,
        '千' | '仟' => 1000,
        _ => return None,
    })
}

fn is_dot(c: char) -> bool {
    matches!(c, '点' | '點' | '.')
}

fn is_neg(c: char) -> bool {
    // 付/富/复是"负"最常见的识别谐音
    matches!(c, '负' | '負' | '付' | '富' | '复' | '覆' | '逆' | '减' | '減' | '-' | '−' | '–')
}

fn is_pos(c: char) -> bool {
    // 证/整/政/郑是"正"的常见识别谐音
    matches!(c, '正' | '证' | '整' | '政' | '郑' | '挣' | '顺' | '順' | '加' | '+')
}

fn is_num_char(c: char) -> bool {
    cn_digit(c).is_some() || mult_val(c).is_some()
}

/// 中文数词求值: "一百二十三" → 123; 不带十百千万的串按逐位拼接: "一二三" → 123
fn cn_int(cs: &[char]) -> Option<f64> {
    if cs.is_empty() {
        return Some(0.0);
    }
    if cs.iter().all(|&c| cn_digit(c).is_some()) {
        let s: String = cs
            .iter()
            .filter_map(|&c| char::from_digit(cn_digit(c)? as u32, 10))
            .collect();
        return s.parse::<f64>().ok();
    }
    let mut total = 0.0f64;
    let mut digit: Option<i64> = None;
    for &c in cs {
        if let Some(d) = cn_digit(c) {
            digit = Some(d);
        } else if let Some(m) = mult_val(c) {
            total += (digit.unwrap_or(1) * m) as f64;
            digit = None;
        } else {
            return None;
        }
    }
    total += digit.unwrap_or(0) as f64;
    Some(total)
}

/// 对一个 token 段求值,返回 (值, 整数部分是否非空)
fn eval_tok(cs: &[char]) -> Option<(f64, bool)> {
    let dot_pos = cs.iter().position(|&c| is_dot(c));
    let (ics, fcs) = match dot_pos {
        Some(p) => (&cs[..p], &cs[p + 1..]),
        None => (cs, &[][..]),
    };
    if ics.is_empty() && fcs.is_empty() {
        return None;
    }
    let iv = cn_int(ics)?;
    // 小数部分逐位拼接
    let mut fv = 0.0f64;
    let mut base = 0.1f64;
    for &c in fcs {
        match cn_digit(c) {
            Some(d) => {
                fv += d as f64 * base;
                base /= 10.0;
            }
            None => return None,
        }
    }
    Some((iv + fv, !ics.is_empty()))
}

/// 常听模式用:文本里必须出现"风/速"类关键词才接受数字,
/// 避免环境说话声误改风速。"风速负三点二"→ -3.2
pub fn parse_wind_gated(text: &str) -> Option<f64> {
    let has_key = text
        .chars()
        .any(|c| matches!(c, '风' | '風' | '速' | '封' | '枫' | '楓'));
    if has_key {
        parse_wind_speech(text)
    } else {
        None
    }
}

/// 语音报角度:必须带"度/角"类关键词,取最后一个数字的绝对值
/// "六十五度"→65,"角度30"→30。和风速关键词错开,互不抢
pub fn parse_angle_speech(text: &str) -> Option<f64> {
    let has_key = text
        .chars()
        .any(|c| matches!(c, '度' | '角' | '肚' | '渡' | '°'));
    if has_key {
        parse_wind_speech(text).map(|v| v.abs())
    } else {
        None
    }
}

pub fn parse_wind_speech(text: &str) -> Option<f64> {
    // 剥掉所有空白:whisper 有时输出 "富三 点 二" 这种带空格的,
    // 空格只会是识别噪声,剥掉不影响符号/数字判断
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '\u{3000}')
        .collect();
    // 切 token: 数字/小数点 的极大连续段
    let mut toks: Vec<(usize, f64, bool)> = Vec::new(); // (起始字符下标, 值, 有整数部分)
    let mut i = 0;
    while i < chars.len() {
        if is_num_char(chars[i]) || is_dot(chars[i]) {
            let start = i;
            while i < chars.len() && (is_num_char(chars[i]) || is_dot(chars[i])) {
                i += 1;
            }
            if let Some((v, has_int)) = eval_tok(&chars[start..i]) {
                toks.push((start, v, has_int));
            }
        } else {
            i += 1;
        }
    }
    // "负3, 点2" 这种被标点拆开的:没有整数部分的 token 并到前一个的整数上
    let mut merged: Vec<(usize, f64)> = Vec::new();
    for (start, v, has_int) in toks {
        if !has_int {
            if let Some(last) = merged.last_mut() {
                *last = (last.0, last.1 + v);
                continue;
            }
        }
        merged.push((start, v));
    }
    let (start, val) = *merged.last()?;
    // 符号:该数字前面最近的正/负词
    let neg = chars[..start]
        .iter()
        .rev()
        .find(|&&c| is_neg(c) || is_pos(c))
        .map(|&c| is_neg(c))
        .unwrap_or(false);
    Some(if neg { -val } else { val })
}

#[cfg(test)]
mod tests {
    use super::{parse_angle_speech, parse_wind_gated, parse_wind_speech};

    #[test]
    fn angle() {
        assert_eq!(parse_angle_speech("六十五度"), Some(65.0));
        assert_eq!(parse_angle_speech("角度30"), Some(30.0));
        assert_eq!(parse_angle_speech("45°"), Some(45.0));
        assert_eq!(parse_angle_speech("85"), None); // 无关键词→不是角度
        assert_eq!(parse_angle_speech("风速5.2"), None); // 不抢风速
    }

    #[test]
    fn gated() {
        // 常听模式:必须有风/速关键词
        assert_eq!(parse_wind_gated("风速负三点二"), Some(-3.2));
        assert_eq!(parse_wind_gated("封速-3.2"), Some(-3.2)); // 谐音也认
        assert_eq!(parse_wind_gated("负三点二"), None); // 没关键词,忽略
        assert_eq!(parse_wind_gated("今天天气不错"), None);
    }

    #[test]
    fn arabic() {
        assert_eq!(parse_wind_speech("正12"), Some(12.0));
        assert_eq!(parse_wind_speech("负123.2"), Some(-123.2));
        assert_eq!(parse_wind_speech("12.5"), Some(12.5));
        assert_eq!(parse_wind_speech("-3.2"), Some(-3.2));
    }

    #[test]
    fn chinese_digits() {
        assert_eq!(parse_wind_speech("正十二"), Some(12.0));
        assert_eq!(parse_wind_speech("负三点二"), Some(-3.2));
        assert_eq!(parse_wind_speech("负一百二十三点二"), Some(-123.2));
        assert_eq!(parse_wind_speech("减二十"), Some(-20.0));
        assert_eq!(parse_wind_speech("逆风零点五"), Some(-0.5));
    }

    #[test]
    fn homophone_and_traditional() {
        // whisper 实际输出:负→付/富,点→點;SenseVoice 输出:负→-,正→证
        assert_eq!(parse_wind_speech("付123.2"), Some(-123.2));
        assert_eq!(parse_wind_speech("富三點二"), Some(-3.2));
        assert_eq!(parse_wind_speech("付5.8"), Some(-5.8));
        assert_eq!(parse_wind_speech("负十二點五"), Some(-12.5));
        assert_eq!(parse_wind_speech("证12。"), Some(12.0));
        assert_eq!(parse_wind_speech("-3.2。"), Some(-3.2));
    }

    #[test]
    fn last_wins() {
        // 说了多个数字,后说的覆盖前面的
        assert_eq!(parse_wind_speech("负5 正8"), Some(8.0));
        assert_eq!(parse_wind_speech("正3.5, 不对, 负二"), Some(-2.0));
    }

    #[test]
    fn no_number() {
        assert_eq!(parse_wind_speech("你好"), None);
        assert_eq!(parse_wind_speech("负"), None);
        assert_eq!(parse_wind_speech(""), None);
    }

    #[test]
    fn spaced_digits() {
        // whisper 偶发在数字间插空格
        assert_eq!(parse_wind_speech("富三 点 二"), Some(-3.2));
        assert_eq!(parse_wind_speech("负 1 2 . 5"), Some(-12.5));
    }

    #[test]
    fn noise_around() {
        assert_eq!(parse_wind_speech("风速负三"), Some(-3.0));
        assert_eq!(parse_wind_speech("呃负3,点2吧"), Some(-3.2));
        assert_eq!(parse_wind_speech("正零"), Some(0.0));
    }
}
