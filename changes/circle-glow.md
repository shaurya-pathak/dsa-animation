bump: minor
Circle objects now render with a circular radial glow halo when highlighted via pulse animation, instead of the rectangular overlay used for other object types. Also fixes a Cairo rendering bug where ctx.arc() without ctx.new_path() would draw an implicit line from the previous text cursor position to the arc start.
