import json
import traceback


class Log:

    def __init__(self):

        self.counter = 0

        # mimic JS environment detection
        try:
            from server.settings import Settings
            self.settings = Settings()
        except Exception:
            try:
                self.settings = g.sSettings
            except Exception:
                # fallback
                self.settings = type("Settings", (), {"log": True})()

    def log(self, o):  # data

        # description,
        # function (function name),
        # number

        message = ""

        if (getattr(self.settings, "log", False) is True or
                o.get("err") is not None or
                o.get("logAnyway") is True):

            self.counter += 1

            try:

                data = ""
                description = o.get("description")
                err = ""

                if o.get("data") is not None:

                    data = json.dumps(o.get("data")) \
                        .replace("\\r\\n", "\r\n") \
                        .replace("{", "\r\n{") + "\r\n\r\n"

                if o.get("err") is not None:

                    description = "catched " + ("programmer " if o.get("perr") else "") + "err"

                    err_obj = o.get("err")
                    err = "".join(
                        traceback.format_exception(type(err_obj), err_obj, err_obj.__traceback__)
                    ) + "\r\n\r\n"

                message = (
                    str(self.counter) +
                    "\r\n\r\n" +
                    str(o.get("function")) + " " +
                    "\r\n\r\n" +
                    str(description) + ": " +
                    "\r\n" +
                    data +
                    err
                )

                if o.get("returnMessage"):

                    return message

                else:

                    print(message)

            except Exception as err:

                print("logging failed: " + str(err))
