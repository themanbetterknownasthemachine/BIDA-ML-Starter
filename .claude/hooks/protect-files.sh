#!/usr/bin/env bash
# PreToolUse-Hook: blockt gefaehrliche Operationen, egal was das Modell entscheidet.
# Eingerichtet in .claude/settings.json unter hooks.PreToolUse. Exit-Code 2 = blockieren.
# Diese Enforcement-Schicht ist fail-closed: ohne funktionierenden Python-Interpreter,
# bei nicht lesbarem Payload oder abgestuerzter Pruefung wird geblockt statt
# stillschweigend durchgelassen.

# Nur ein echtes Python zaehlt: Der Windows-Store-Platzhalter oder eine leere bzw. fremde
# Datei liefern nicht die erwartete Ausgabe (eine leere Datei wuerde Git Bash sonst als
# leeres Skript mit Exit 0 ausfuehren, und die Pruefung liefe ins Leere).
is_python() {
  [ "$("$1" -c "import sys; sys.stdout.write('ok')" 2>/dev/null)" = "ok" ]
}

py_bin=""
for c in python3 python py; do
  if is_python "$c"; then
    py_bin="$c"
    break
  fi
done
# Rueckfallebene fuer Geschaeftsnotebooks ohne Python im PATH (dort liegt oft nur der
# Windows-Store-Platzhalter, der die Probe oben nicht besteht): das portable Team-WinPython
# im Benutzerordner, zuerst C:\Users\<kuerzel>\Documents\WPy64-*, dann C:\Users\<kuerzel>\WPy64-*.
if [ -z "$py_bin" ]; then
  user_dir="${USERPROFILE:-$HOME}"
  if [ -n "$user_dir" ] && command -v cygpath >/dev/null 2>&1; then
    user_dir="$(cygpath -u "$user_dir")"
  fi
  for c in "$user_dir"/Documents/WPy64-*/python/python.exe "$user_dir"/WPy64-*/python/python.exe; do
    if [ -f "$c" ] && is_python "$c"; then
      py_bin="$c"
      break
    fi
  done
fi
if [ -z "$py_bin" ]; then
  echo "Blockiert durch protect-files.sh: kein Python-Interpreter gefunden (weder im PATH noch als WinPython unter Documents), Pruefung nicht moeglich." >&2
  exit 2
fi

reason="$("$py_bin" "$(dirname "$0")/_check.py")"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "Blockiert durch protect-files.sh: Pruefung abgebrochen (Exit ${rc}), Aktion nicht freigegeben." >&2
  exit 2
fi
if [ -n "$reason" ]; then
  echo "Blockiert durch protect-files.sh: ${reason} ist ohne explizite Freigabe gesperrt." >&2
  exit 2
fi
exit 0
