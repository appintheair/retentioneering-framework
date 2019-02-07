from IPython.display import HTML, display

_end = '_end_'


def _make_trie_of_events(data):
    events = data.event_name.unique()
    root = dict()
    for event in events:
        current_dict = root
        for e in event.split('_'):
            current_dict = current_dict.setdefault(e, {})
        current_dict[_end] = event
    return root


def _add_checkbox(cur_dict, cur_prefix='', text='', is_checked=True):
    _end = '_end_'
    for key in cur_dict:
        if key == _end:
            cur_prefix += '-0'
            text += '''
            <li class="last">
                <input type="checkbox" name="{check_id}" id="{check_id}" {is_check}>
                <label for="{check_id}" class="no-children custom-{is_check}">{check_name}</label>
            </li>
            '''.format(check_id=cur_prefix, check_name=cur_dict[key],
                       is_check='checked' if is_checked else 'unchecked')
            return text
        else:
            cur_prefix += ('-' + key if cur_prefix else key)
            text += '''
            <li>
                <input type="checkbox" name="{check_id}" id="{check_id}" {is_check}>
                <label for="{check_id}" class="custom-{is_check}">{check_name}</label>
                <ul>
            '''.format(check_id=cur_prefix, check_name=key,
                       is_check='checked' if is_checked else 'unchecked')
            text = _add_checkbox(cur_dict[key], cur_prefix=cur_prefix, text=text, is_checked=is_checked)
            text += '''
                </ul>
            </li>
            '''
    return text


def print_checkboxes(data, checkbox_id='1', is_checked=True):
    display(HTML('''
    <!DOCTYPE html>
    <html>

    <head>
      <meta charset='UTF-8'>

    <script>
    $(function() {

      $('input[type="checkbox"]').change(checkboxChanged);

      function checkboxChanged() {
        var $this = $(this),
            checked = $this.prop("checked"),
            container = $this.parent(),
            siblings = container.siblings();

        container.find('input[type="checkbox"]')
        .prop({
            indeterminate: false,
            checked: checked
        })
        .siblings('label')
        .removeClass('custom-checked custom-unchecked custom-indeterminate')
        .addClass(checked ? 'custom-checked' : 'custom-unchecked');

        checkSiblings(container, checked);
      }

      function checkSiblings($el, checked) {
        var parent = $el.parent().parent(),
            all = true,
            indeterminate = false;

        $el.siblings().each(function() {
          return all = ($(this).children('input[type="checkbox"]').prop("checked") === checked);
        });

        if (all && checked) {
          parent.children('input[type="checkbox"]')
          .prop({
              indeterminate: false,
              checked: checked
          })
          .siblings('label')
          .removeClass('custom-checked custom-unchecked custom-indeterminate')
          .addClass(checked ? 'custom-checked' : 'custom-unchecked');

          checkSiblings(parent, checked);
        } 
        else if (all && !checked) {
          indeterminate = parent.find('input[type="checkbox"]:checked').length > 0;

          parent.children('input[type="checkbox"]')
          .prop("checked", checked)
          .prop("indeterminate", indeterminate)
          .siblings('label')
          .removeClass('custom-checked custom-unchecked custom-indeterminate')
          .addClass(indeterminate ? 'custom-indeterminate' : (checked ? 'custom-checked' : 'custom-unchecked'));

          checkSiblings(parent, checked);
        } 
        else {
          $el.parents("li").children('input[type="checkbox"]')
          .prop({
              indeterminate: true,
              checked: false
          })
          .siblings('label')
          .removeClass('custom-checked custom-unchecked custom-indeterminate')
          .addClass('custom-indeterminate');
        }
      }
    });
    function buttonPressed() {
        var result = [];
        $("ul.treeview''' + checkbox_id + '''").find('label[class="no-children custom-checked"]').each(function () {
            result.push($(this).context.textContent);
        });
        IPython.notebook.kernel.execute('result_filter='+JSON.stringify(result));
    }
    buttonPressed()
    </script>

    </head>

    <body>

    <div id="page-wrap">

         <h2>Graph event filter</h2>

         <ul class="treeview''' + checkbox_id + '''">
    ''' + _add_checkbox(_make_trie_of_events(data), text='', is_checked=is_checked) + '''        
        </ul>

      </div>

      <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"></script>
      <button onclick="buttonPressed()">Apply filters</button>
    </body>


    </html>'''))
