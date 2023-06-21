def any_keyword_in_string(string, keywords):
  for keyword in keywords:
    if keyword in string:
      return True
  return False